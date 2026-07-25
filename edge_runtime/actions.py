"""Normalized ONNX action to physical velocity conversion."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple


ACTION_ORDER = ("boom", "stick", "bucket", "swing")


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
