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


def dual_rate_slew_limited_normalized_action(
    target: Sequence[float],
    previous: Sequence[float],
    *,
    startup_pending: Sequence[bool],
    elapsed_s: float,
    startup_rate_per_s: float,
    steady_rate_per_s: float,
) -> tuple[
    Tuple[float, float, float, float],
    tuple[bool, bool, bool, bool],
]:
    """Use a faster first ramp while preserving steady/reversal smoothing.

    Each axis remains in startup until it reaches its first non-zero target.
    A reduction or reversal ends startup for that axis immediately, so later
    changes always use the steady rate.
    """
    if len(startup_pending) != 4 or not all(
        isinstance(value, bool) for value in startup_pending
    ):
        raise ValueError("startup_pending must contain four booleans")
    steady = slew_limited_normalized_action(
        target,
        previous,
        elapsed_s=elapsed_s,
        max_rate_per_s=steady_rate_per_s,
    )
    startup = slew_limited_normalized_action(
        target,
        previous,
        elapsed_s=elapsed_s,
        max_rate_per_s=startup_rate_per_s,
    )
    target_values = tuple(float(value) for value in target)
    previous_values = tuple(float(value) for value in previous)
    action = []
    pending_after = []
    for desired, current, pending, startup_value, steady_value in zip(
        target_values,
        previous_values,
        startup_pending,
        startup,
        steady,
    ):
        increasing_from_zero = (
            desired != 0.0
            and (
                current == 0.0
                or math.copysign(1.0, desired)
                == math.copysign(1.0, current)
            )
            and abs(desired) >= abs(current)
        )
        use_startup = pending and increasing_from_zero
        next_value = startup_value if use_startup else steady_value
        action.append(next_value)
        if not pending:
            pending_after.append(False)
        elif desired == 0.0:
            pending_after.append(True)
        elif not use_startup:
            pending_after.append(False)
        else:
            pending_after.append(not math.isclose(next_value, desired, abs_tol=1e-12))
    return (
        tuple(action),
        tuple(pending_after),
    )  # type: ignore[return-value]


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
