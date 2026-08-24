"""Strict local policy-worker protocol for the Resident Mission Runtime."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .resident_motion import ACTION_ORDER, ControlMode, MotionCandidate


CANDIDATE_SCHEMA_VERSION = "resident_policy_candidate.v1"
MAX_CANDIDATE_BYTES = 4096
UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_FIELDS = {
    "schema_version",
    "source",
    "control_generation",
    "mode",
    "action_order",
    "action",
    "created_monotonic_ns",
    "valid_until_monotonic_ns",
}


def encode_motion_candidate(candidate: MotionCandidate) -> bytes:
    if not isinstance(candidate, MotionCandidate):
        raise ValueError("candidate must be a MotionCandidate")
    payload = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "source": candidate.source,
        "control_generation": candidate.generation,
        "mode": candidate.mode.value,
        "action_order": list(ACTION_ORDER),
        "action": list(candidate.action),
        "created_monotonic_ns": candidate.created_monotonic_ns,
        "valid_until_monotonic_ns": candidate.valid_until_monotonic_ns,
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate cannot be encoded as finite JSON") from exc
    if len(encoded) > MAX_CANDIDATE_BYTES:
        raise ValueError("candidate payload exceeds the local protocol limit")
    return encoded


def decode_motion_candidate(payload: bytes) -> MotionCandidate:
    if not isinstance(payload, bytes):
        raise ValueError("candidate payload must be bytes")
    if not payload or len(payload) > MAX_CANDIDATE_BYTES:
        raise ValueError("candidate payload size is invalid")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("candidate payload is not strict finite JSON") from exc
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("candidate payload fields are invalid")
    if value["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate schema_version is unsupported")
    if value["action_order"] != list(ACTION_ORDER):
        raise ValueError("candidate action_order must be canonical")

    source = _text("source", value["source"])
    generation = _nonnegative_integer(
        "control_generation",
        value["control_generation"],
        maximum=UINT64_MAX,
    )
    created = _nonnegative_integer(
        "created_monotonic_ns", value["created_monotonic_ns"]
    )
    valid_until = _nonnegative_integer(
        "valid_until_monotonic_ns", value["valid_until_monotonic_ns"]
    )
    try:
        mode = ControlMode(value["mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate mode is invalid") from exc
    action_value = value["action"]
    if not isinstance(action_value, list) or len(action_value) != len(ACTION_ORDER):
        raise ValueError("candidate action must contain four values")
    action = tuple(_finite_number(f"action[{index}]", item) for index, item in enumerate(action_value))
    return MotionCandidate(
        source=source,
        generation=generation,
        mode=mode,
        action=action,
        created_monotonic_ns=created,
        valid_until_monotonic_ns=valid_until,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field is forbidden: {key}")
        value[key] = item
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate {name} must be a non-empty string")
    return value.strip()


def _nonnegative_integer(
    name: str,
    value: Any,
    *,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"candidate {name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"candidate {name} exceeds its allowed range")
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate {name} must be finite")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate {name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"candidate {name} must be finite")
    return number
