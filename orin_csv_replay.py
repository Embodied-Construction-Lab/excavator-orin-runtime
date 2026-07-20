#!/usr/bin/env python3
"""Replay physical-velocity CSV commands through the local Orin action relay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


ACTION_ORDER = ("boom", "stick", "bucket", "swing")
VELOCITY_COLUMNS = (
    "boom_v_ref_mps",
    "stick_v_ref_mps",
    "bucket_v_ref_mps",
    "swing_v_ref_radps",
)
REQUIRED_AUTHORIZATION = "ALLOW_CSV_REPLAY"
DEFAULT_MACHINE_PROFILE = Path(__file__).with_name("machine_profile.json")


class ReplayValidationError(ValueError):
    """The replay artifact is unsafe or incompatible with the active contract."""


@dataclass(frozen=True)
class ReplaySample:
    timestamp_s: float
    action: tuple[float, float, float, float]


def is_execution_authorized(value: str | None) -> bool:
    return value == REQUIRED_AUTHORIZATION


def _load_limits(machine_profile_path: Path) -> tuple[tuple[float, float], ...]:
    try:
        profile = json.loads(machine_profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"cannot read machine profile: {exc}") from exc
    if tuple(profile.get("action_order", ())) != ACTION_ORDER:
        raise ReplayValidationError(
            f"machine profile action_order must be {list(ACTION_ORDER)}"
        )
    try:
        limits = tuple(
            (
                float(profile["actuators"][name]["max_speed_positive"]),
                float(profile["actuators"][name]["max_speed_negative"]),
            )
            for name in ACTION_ORDER
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayValidationError("machine profile velocity limits are invalid") from exc
    if any(
        not math.isfinite(value) or value <= 0.0
        for pair in limits
        for value in pair
    ):
        raise ReplayValidationError("machine profile velocity limits must be positive and finite")
    return limits


def load_replay(
    csv_path: Path,
    machine_profile_path: Path,
    *,
    max_duration_s: float,
) -> tuple[ReplaySample, ...]:
    """Load physical commands and reject incompatible, unbounded or partial CSVs."""
    if not math.isfinite(max_duration_s) or max_duration_s <= 0.0:
        raise ReplayValidationError("max_duration_s must be positive and finite")
    limits = _load_limits(machine_profile_path)
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ReplayValidationError(f"cannot read replay CSV: {exc}") from exc
    lines = text.splitlines()
    comments = tuple(line.strip() for line in lines if line.startswith("#"))
    if "# profile_action_order=boom|stick|bucket|swing" not in comments:
        raise ReplayValidationError("CSV action-order provenance is missing or incompatible")
    if not any("boom/stick/bucket velocity=m/s" in line for line in comments):
        raise ReplayValidationError("CSV physical velocity unit provenance is missing")
    data_lines = [line for line in lines if line and not line.startswith("#")]
    if not data_lines:
        raise ReplayValidationError("CSV contains no table")
    try:
        reader = csv.DictReader(data_lines)
        required = {"timestamp_s", *VELOCITY_COLUMNS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ReplayValidationError(f"CSV must contain columns {sorted(required)}")
        samples = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp_s = float(row["timestamp_s"])
                action = tuple(float(row[column]) for column in VELOCITY_COLUMNS)
            except (TypeError, ValueError) as exc:
                raise ReplayValidationError(
                    f"CSV row {row_number} contains a non-numeric command"
                ) from exc
            if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
                raise ReplayValidationError(f"CSV row {row_number} has invalid timestamp")
            if not all(math.isfinite(value) for value in action):
                raise ReplayValidationError(f"CSV row {row_number} has non-finite velocity")
            for name, value, (positive, negative) in zip(
                ACTION_ORDER, action, limits, strict=True
            ):
                limit = positive if value >= 0.0 else negative
                magnitude = abs(value)
                if magnitude > limit and not math.isclose(
                    magnitude, limit, rel_tol=1e-6, abs_tol=1e-9
                ):
                    raise ReplayValidationError(
                        f"CSV row {row_number} {name} velocity {value} exceeds {limit}"
                    )
            samples.append(ReplaySample(timestamp_s, action))
    except csv.Error as exc:
        raise ReplayValidationError(f"invalid CSV: {exc}") from exc
    if not samples:
        raise ReplayValidationError("CSV contains no samples")
    for previous, current in zip(samples, samples[1:], strict=False):
        if current.timestamp_s < previous.timestamp_s:
            raise ReplayValidationError("CSV timestamps must not decrease")
    duration_s = samples[-1].timestamp_s - samples[0].timestamp_s
    if duration_s > max_duration_s:
        raise ReplayValidationError(
            f"CSV duration {duration_s:.3f}s exceeds limit {max_duration_s:.3f}s"
        )
    if any(samples[-1].action):
        raise ReplayValidationError("CSV must end with an explicit zero command")
    return tuple(samples)


def make_policy_action(
    *,
    sequence: int,
    action: Sequence[float],
    stamp_ms: int,
    valid_for_ms: int,
) -> dict:
    """Build the existing compatibility packet without changing physical values."""
    if len(action) != len(ACTION_ORDER):
        raise ValueError("action must contain four values")
    if valid_for_ms <= 0:
        raise ValueError("valid_for_ms must be positive")
    return {
        "type": "policy_action",
        "schema_version": "1.0",
        "seq": int(sequence),
        "stamp_ms": int(stamp_ms),
        "action_order": list(ACTION_ORDER),
        "action": [float(value) for value in action],
        # Compatibility label: the values above are physical velocities.
        "action_type": "normalized_velocity_command",
        "valid_for_ms": int(valid_for_ms),
    }


def replay_samples(
    samples: Sequence[ReplaySample],
    send_packet: Callable[[dict], None],
    *,
    valid_for_ms: int,
    zero_tail_count: int,
    zero_interval_s: float,
    max_lag_s: float,
    sequence_start: int,
    monotonic: Callable[[], float] = time.monotonic,
    wall_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Send one packet per CSV row and always append fresh terminal zeros."""
    if not samples:
        raise ValueError("samples must not be empty")
    if valid_for_ms <= 0 or zero_tail_count <= 0:
        raise ValueError("valid_for_ms and zero_tail_count must be positive")
    if zero_interval_s < 0.0 or max_lag_s < 0.0:
        raise ValueError("timing limits must not be negative")
    sequence = int(sequence_start)
    sent_samples = 0
    started_at = monotonic()
    source_start = samples[0].timestamp_s
    try:
        for sample in samples:
            deadline = started_at + sample.timestamp_s - source_start
            remaining = deadline - monotonic()
            if remaining > 0.0:
                sleep(remaining)
            lag_s = monotonic() - deadline
            if lag_s > max_lag_s:
                raise RuntimeError(
                    f"replay scheduler lag {lag_s:.3f}s exceeds {max_lag_s:.3f}s"
                )
            send_packet(
                make_policy_action(
                    sequence=sequence,
                    action=sample.action,
                    stamp_ms=wall_ms(),
                    valid_for_ms=valid_for_ms,
                )
            )
            sequence += 1
            sent_samples += 1
    finally:
        zero = (0.0, 0.0, 0.0, 0.0)
        for index in range(zero_tail_count):
            send_packet(
                make_policy_action(
                    sequence=sequence,
                    action=zero,
                    stamp_ms=wall_ms(),
                    valid_for_ms=valid_for_ms,
                )
            )
            sequence += 1
            if index + 1 < zero_tail_count:
                sleep(zero_interval_s)
    return sent_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and replay physical-velocity CSV rows through a local "
            "orin_state_sender action relay. Without the exact authorization "
            "token this command only performs a dry-run preflight."
        )
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--machine-profile", type=Path, default=DEFAULT_MACHINE_PROFILE
    )
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=18082)
    parser.add_argument("--valid-for-ms", type=int, default=100)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--max-lag-ms", type=float, default=50.0)
    parser.add_argument("--zero-tail-count", type=int, default=10)
    parser.add_argument("--zero-interval-ms", type=float, default=50.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument(
        "--motion-authorization",
        help=f"exact value required for UDP send: {REQUIRED_AUTHORIZATION}",
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if not 1 <= args.target_port <= 65535:
        raise ReplayValidationError("target_port must be in 1..65535")
    if args.valid_for_ms <= 0 or args.valid_for_ms > 1000:
        raise ReplayValidationError("valid_for_ms must be in 1..1000")
    if args.max_lag_ms < 0.0 or args.zero_interval_ms < 0.0:
        raise ReplayValidationError("timing arguments must not be negative")
    if args.zero_tail_count <= 0 or args.print_every < 0:
        raise ReplayValidationError("zero_tail_count must be positive and print_every non-negative")


def _summary(samples: Sequence[ReplaySample]) -> str:
    duration_s = samples[-1].timestamp_s - samples[0].timestamp_s
    peaks = [max(abs(sample.action[index]) for sample in samples) for index in range(4)]
    return (
        f"samples={len(samples)} duration_s={duration_s:.3f} "
        + " ".join(
            f"{name}_peak={peak:.9g}" for name, peak in zip(ACTION_ORDER, peaks, strict=True)
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    samples = load_replay(
        args.csv_path,
        args.machine_profile,
        max_duration_s=args.max_duration_s,
    )
    csv_sha256 = hashlib.sha256(args.csv_path.read_bytes()).hexdigest()
    profile_sha256 = hashlib.sha256(args.machine_profile.read_bytes()).hexdigest()
    print(_summary(samples))
    print(f"csv_sha256={csv_sha256}")
    print(f"machine_profile_sha256={profile_sha256}")
    print("action_order=boom,stick,bucket,swing values=physical_velocity")
    if not is_execution_authorized(args.motion_authorization):
        print("PREVIEW ONLY: no UDP packets sent")
        print(
            "To execute, add: "
            f"--motion-authorization {REQUIRED_AUTHORIZATION}"
        )
        return 0

    resolved_host = socket.gethostbyname(args.target_host)
    if not resolved_host.startswith("127."):
        raise ReplayValidationError(
            "CSV replay target must be loopback; run it on the Orin beside orin_state_sender"
        )
    address = (resolved_host, args.target_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0

    def send_packet(packet: dict) -> None:
        nonlocal sent
        payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(payload, address)
        sent += 1
        if args.print_every and sent % args.print_every == 0:
            print(
                f"sent={sent} seq={packet['seq']} action={packet['action']}"
            )

    try:
        replayed = replay_samples(
            samples,
            send_packet,
            valid_for_ms=args.valid_for_ms,
            zero_tail_count=args.zero_tail_count,
            zero_interval_s=args.zero_interval_ms / 1000.0,
            max_lag_s=args.max_lag_ms / 1000.0,
            sequence_start=time.time_ns() // 1_000_000,
        )
    except KeyboardInterrupt:
        print("Interrupted; terminal zero tail sent")
        return 130
    finally:
        sock.close()
    print(f"REPLAY COMPLETE: samples={replayed}, datagrams={sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
