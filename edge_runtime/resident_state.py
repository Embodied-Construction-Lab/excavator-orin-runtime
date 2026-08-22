"""Strict local ACT state protocol for the Resident Mission Runtime.

The protocol contains only policy-observation and motion-gating evidence.  It
does not choose or own a transport; callers may carry the encoded bytes over a
local pipe, socket, shared-memory queue, or an in-process seam.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


RESIDENT_STATE_SCHEMA_VERSION = "resident_act_state.v1"
MAX_RESIDENT_STATE_BYTES = 4096
UINT32_MAX = 0xFFFFFFFF
UINT64_MAX = 0xFFFFFFFFFFFFFFFF

# Authoritative ``observation.state`` names and order from excavator-il's
# LeRobot data contract.  ``arm`` is the joint-angle name for the physical
# stick degree of freedom; it must not be silently renamed or reordered.
ACT_STATE_NAMES = (
    "boom_pos_m",
    "stick_pos_m",
    "bucket_pos_m",
    "boom_vel_mps",
    "stick_vel_mps",
    "bucket_vel_mps",
    "boom_angle_rad",
    "arm_angle_rad",
    "bucket_angle_rad",
    "swing_angle_rad",
    "swing_vel_radps",
)

_FIELDS = frozenset(
    {
        "schema_version",
        "state_names",
        "state",
        "receive_monotonic_ns",
        "state_monotonic_ns",
        "control_seq",
        "sensor_seq",
        "sensor_is_new",
        "control_enabled",
        "estop",
        "rs485_ok",
        "dwj_ok",
        "imu_ok",
        "sensor_valid",
        "stm32_alive",
        "fault_flags",
        "control_generation",
    }
)


@dataclass(frozen=True)
class ResidentActState:
    """One immutable, named 11D ACT observation plus gating evidence."""

    state: tuple[float, ...]
    receive_monotonic_ns: int
    state_monotonic_ns: int
    control_seq: int
    sensor_seq: int
    sensor_is_new: bool
    control_enabled: bool
    estop: bool
    rs485_ok: bool
    dwj_ok: bool
    imu_ok: bool
    sensor_valid: bool
    stm32_alive: bool
    fault_flags: int
    control_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, tuple) or len(self.state) != len(ACT_STATE_NAMES):
            raise ValueError("resident ACT state must be the canonical 11-value tuple")
        state = tuple(
            _finite_number(f"state[{index}]", value)
            for index, value in enumerate(self.state)
        )
        receive_ns = _uint64("receive_monotonic_ns", self.receive_monotonic_ns)
        state_ns = _uint64("state_monotonic_ns", self.state_monotonic_ns)
        if state_ns > receive_ns:
            raise ValueError("state_monotonic_ns must not exceed receive_monotonic_ns")
        _uint32("control_seq", self.control_seq)
        _uint32("sensor_seq", self.sensor_seq)
        _uint32("fault_flags", self.fault_flags)
        _uint64("control_generation", self.control_generation)
        for name in (
            "sensor_is_new",
            "control_enabled",
            "estop",
            "rs485_ok",
            "dwj_ok",
            "imu_ok",
            "sensor_valid",
            "stm32_alive",
        ):
            _boolean(name, getattr(self, name))
        object.__setattr__(self, "state", state)


def encode_resident_state(frame: ResidentActState) -> bytes:
    """Encode one validated state frame as finite, deterministic JSON bytes."""

    if not isinstance(frame, ResidentActState):
        raise ValueError("resident state must be a ResidentActState")
    payload = {
        "schema_version": RESIDENT_STATE_SCHEMA_VERSION,
        "state_names": list(ACT_STATE_NAMES),
        "state": list(frame.state),
        "receive_monotonic_ns": frame.receive_monotonic_ns,
        "state_monotonic_ns": frame.state_monotonic_ns,
        "control_seq": frame.control_seq,
        "sensor_seq": frame.sensor_seq,
        "sensor_is_new": frame.sensor_is_new,
        "control_enabled": frame.control_enabled,
        "estop": frame.estop,
        "rs485_ok": frame.rs485_ok,
        "dwj_ok": frame.dwj_ok,
        "imu_ok": frame.imu_ok,
        "sensor_valid": frame.sensor_valid,
        "stm32_alive": frame.stm32_alive,
        "fault_flags": frame.fault_flags,
        "control_generation": frame.control_generation,
    }
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("resident state cannot be encoded as finite JSON") from exc
    if len(encoded) > MAX_RESIDENT_STATE_BYTES:
        raise ValueError("resident state exceeds the local protocol limit")
    return encoded


def decode_resident_state(payload: bytes) -> ResidentActState:
    """Decode one strict, finite, exact-field resident ACT state frame."""

    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > MAX_RESIDENT_STATE_BYTES
    ):
        raise ValueError("resident state payload size is invalid")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("resident state is not strict finite JSON") from exc
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("resident state fields are invalid")
    if value["schema_version"] != RESIDENT_STATE_SCHEMA_VERSION:
        raise ValueError("resident state schema_version is unsupported")
    if value["state_names"] != list(ACT_STATE_NAMES):
        raise ValueError("resident state_names must match the canonical ACT order")
    raw_state = value["state"]
    if not isinstance(raw_state, list) or len(raw_state) != len(ACT_STATE_NAMES):
        raise ValueError("resident ACT state must contain exactly 11 values")
    return ResidentActState(
        state=tuple(
            _finite_number(f"state[{index}]", item)
            for index, item in enumerate(raw_state)
        ),
        receive_monotonic_ns=_uint64(
            "receive_monotonic_ns", value["receive_monotonic_ns"]
        ),
        state_monotonic_ns=_uint64(
            "state_monotonic_ns", value["state_monotonic_ns"]
        ),
        control_seq=_uint32("control_seq", value["control_seq"]),
        sensor_seq=_uint32("sensor_seq", value["sensor_seq"]),
        sensor_is_new=_boolean("sensor_is_new", value["sensor_is_new"]),
        control_enabled=_boolean("control_enabled", value["control_enabled"]),
        estop=_boolean("estop", value["estop"]),
        rs485_ok=_boolean("rs485_ok", value["rs485_ok"]),
        dwj_ok=_boolean("dwj_ok", value["dwj_ok"]),
        imu_ok=_boolean("imu_ok", value["imu_ok"]),
        sensor_valid=_boolean("sensor_valid", value["sensor_valid"]),
        stm32_alive=_boolean("stm32_alive", value["stm32_alive"]),
        fault_flags=_uint32("fault_flags", value["fault_flags"]),
        control_generation=_uint64(
            "control_generation", value["control_generation"]
        ),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field is forbidden: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"resident state {name} must be boolean")
    return value


def _uint32(name: str, value: Any) -> int:
    return _bounded_integer(name, value, maximum=UINT32_MAX)


def _uint64(name: str, value: Any) -> int:
    return _bounded_integer(name, value, maximum=UINT64_MAX)


def _bounded_integer(name: str, value: Any, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"resident state {name} is outside its integer range")
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resident state {name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"resident state {name} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"resident state {name} must be a finite number")
    return number
