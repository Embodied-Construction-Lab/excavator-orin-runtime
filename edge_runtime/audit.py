"""Summarize Orin edge JSONL audit logs for Shadow and control diagnosis."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def load_audit_records(path: Path) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("cannot read audit log %s: %s" % (path, exc)) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "audit log line %d is invalid JSON: %s" % (line_number, exc)
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError("audit log line %d is not an object" % line_number)
        records.append(value)
    if not records:
        raise ValueError("audit log contains no records")
    return records


def summarize_audit_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty audit record sequence")
    statuses = Counter(str(record.get("status", "missing")) for record in records)
    source_sequences = _finite_integers(records, "source_seq")
    source_stamps = _finite_numbers(records, "source_stamp_ms")
    runtime_times = _finite_numbers(records, "runtime_monotonic_s")
    progress = _finite_numbers(records, "episode_progress")
    physical_actions = [
        record.get("physical_action")
        for record in records
        if _is_action(record.get("physical_action"))
    ]
    consecutive_rejections = _finite_integers(records, "consecutive_rejections")
    return {
        "record_count": len(records),
        "modes": sorted({str(record.get("mode", "missing")) for record in records}),
        "status_counts": dict(statuses),
        "source_sequence": _sequence_summary(source_sequences),
        "state_input_hz": _frequency_hz(source_stamps, scale=1000.0),
        "inference_loop_hz": _frequency_hz(runtime_times, scale=1.0),
        "inference_ms": _distribution(_finite_numbers(records, "inference_ms")),
        "loop_elapsed_ms": _distribution(
            _finite_numbers(records, "loop_elapsed_ms")
        ),
        "raw_action_delta_abs": _action_delta_summary(
            records,
            "normalized_action",
        ),
        "commanded_action_delta_abs": _action_delta_summary(
            records,
            "commanded_normalized_action",
        ),
        "progress": {
            "first": progress[0] if progress else None,
            "last": progress[-1] if progress else None,
            "monotonic_violations": sum(
                current < previous
                for previous, current in zip(progress, progress[1:])
            ),
        },
        "max_consecutive_rejections": (
            max(consecutive_rejections) if consecutive_rejections else 0
        ),
        "timeout_seen": statuses["TIMEOUT"] > 0,
        "completed_seen": statuses["COMPLETED"] > 0
        or statuses["completed"] > 0,
        "final_physical_action_zero": (
            all(float(value) == 0.0 for value in physical_actions[-1])
            if physical_actions
            else None
        ),
    }


def _action_delta_summary(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> Dict[str, Any]:
    names = ("boom", "stick", "bucket", "swing")
    deltas: List[List[float]] = [[] for _ in names]
    reversals = [0 for _ in names]
    previous_record: Mapping[str, Any] | None = None
    previous_action: Sequence[float] | None = None
    for record in records:
        action = record.get(field)
        if not _is_action(action):
            previous_record = None
            previous_action = None
            continue
        if (
            previous_record is not None
            and previous_action is not None
            and _records_are_contiguous(previous_record, record)
        ):
            for index, (previous, current) in enumerate(
                zip(previous_action, action)
            ):
                before = float(previous)
                after = float(current)
                deltas[index].append(abs(after - before))
                if before * after < 0.0:
                    reversals[index] += 1
        previous_record = record
        previous_action = action
    result = {}
    for index, name in enumerate(names):
        distribution = _distribution(deltas[index])
        distribution["sign_reversals"] = reversals[index]
        result[name] = distribution
    return result


def _records_are_contiguous(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    if previous.get("schema_version") != current.get("schema_version"):
        return False
    previous_sequence = previous.get("source_seq")
    current_sequence = current.get("source_seq")
    if (
        isinstance(previous_sequence, bool)
        or not isinstance(previous_sequence, int)
        or isinstance(current_sequence, bool)
        or not isinstance(current_sequence, int)
        or current_sequence <= previous_sequence
    ):
        return False
    previous_time = previous.get("runtime_monotonic_s")
    current_time = current.get("runtime_monotonic_s")
    if (
        isinstance(previous_time, bool)
        or not isinstance(previous_time, (int, float))
        or isinstance(current_time, bool)
        or not isinstance(current_time, (int, float))
    ):
        return False
    elapsed = float(current_time) - float(previous_time)
    return math.isfinite(elapsed) and 0.0 <= elapsed <= 2.0


def _finite_numbers(
    records: Iterable[Mapping[str, Any]],
    field: str,
) -> List[float]:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        converted = float(value)
        if math.isfinite(converted):
            values.append(converted)
    return values


def _finite_integers(
    records: Iterable[Mapping[str, Any]],
    field: str,
) -> List[int]:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        values.append(value)
    return values


def _sequence_summary(values: Sequence[int]) -> Dict[str, Any]:
    differences = [
        current - previous for previous, current in zip(values, values[1:])
    ]
    return {
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "missing_count": sum(max(difference - 1, 0) for difference in differences),
        "non_increasing_count": sum(difference <= 0 for difference in differences),
    }


def _frequency_hz(values: Sequence[float], *, scale: float) -> Any:
    if len(values) < 2:
        return None
    elapsed = values[-1] - values[0]
    if elapsed <= 0.0:
        return None
    return scale * (len(values) - 1) / elapsed


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _is_action(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_path", type=Path)
    args = parser.parse_args(argv)
    summary = summarize_audit_records(load_audit_records(args.audit_path))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
