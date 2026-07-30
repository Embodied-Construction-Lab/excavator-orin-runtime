"""Normalized ONNX action to physical velocity conversion."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple


ACTION_ORDER = ("boom", "stick", "bucket", "swing")


def slew_limited_normalized_action(
    target: Sequence[float],
    previous: Sequence[float],
    *,
    elapsed_s: float,
    max_rate_per_s: float,
) -> Tuple[float, float, float, float]:
    """Approach a normalized policy target without an instantaneous reversal."""
    if len(target) != 4 or len(previous) != 4:
        raise ValueError("slew-limited actions must contain four values")
    elapsed = float(elapsed_s)
    rate = float(max_rate_per_s)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("elapsed_s must be finite and nonnegative")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("max_rate_per_s must be finite and positive")
    target_values = tuple(float(value) for value in target)
    previous_values = tuple(float(value) for value in previous)
    if not all(
        math.isfinite(value) for value in target_values + previous_values
    ):
        raise ValueError("slew-limited actions must be finite")
    max_delta = rate * elapsed
    result = tuple(
        current
        + max(-max_delta, min(max_delta, desired - current))
        for desired, current in zip(target_values, previous_values)
    )
    return result  # type: ignore[return-value]


def physical_velocity_from_normalized(
    action: Sequence[float],
    machine_profile: Mapping[str, Any],
) -> Tuple[float, float, float, float]:
    if len(action) != 4:
        raise ValueError("policy action must contain four values")
    if tuple(machine_profile.get("action_order", ())) != ACTION_ORDER:
        raise ValueError("machine profile action_order is invalid")
    result = []
    for name, raw_value in zip(ACTION_ORDER, action):
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("policy action contains a non-finite value")
        normalized = max(-1.0, min(1.0, value))
        actuator = machine_profile["actuators"][name]
        deadzone_name = (
            "command_deadzone_positive_normalized"
            if normalized >= 0.0
            else "command_deadzone_negative_normalized"
        )
        deadzone = float(actuator.get(deadzone_name, 0.0))
        if abs(normalized) <= deadzone:
            result.append(0.0)
            continue
        speed = float(
            actuator["max_speed_positive"]
            if normalized >= 0.0
            else actuator["max_speed_negative"]
        )
        result.append(normalized * speed)
    return tuple(result)  # type: ignore[return-value]
