"""Unity-compatible 38-dimensional observation construction on Orin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


ACTION_ORDER = ("boom", "stick", "bucket", "swing")


@dataclass(frozen=True)
class BucketTipObservation:
    position_m: Tuple[float, float, float]
    pitch_rad: float
    stamp_ms: int


def position_to_unity(position_m: Sequence[float]) -> Tuple[float, float, float]:
    if len(position_m) != 3:
        raise ValueError("ROS position must contain exactly three values")
    x_forward, y_left, z_up = (float(value) for value in position_m)
    return -y_left, z_up, x_forward


def bucket_pitch_from_ros_pose(
    orientation_xyzw: Sequence[float],
    *,
    swing_joint_rad: float,
) -> float:
    if len(orientation_xyzw) != 4:
        raise ValueError("ROS quaternion must contain exactly four values")
    quaternion = tuple(float(value) for value in orientation_xyzw)
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-9:
        raise ValueError("ROS quaternion cannot be zero")
    qx, qy, qz, qw = (value / norm for value in quaternion)
    bucket_x = 1.0 - 2.0 * (qy * qy + qz * qz)
    bucket_y = 2.0 * (qx * qy + qw * qz)
    bucket_z = 2.0 * (qx * qz - qw * qy)
    swing = float(swing_joint_rad)
    if not math.isfinite(swing):
        raise ValueError("swing_joint_rad must be finite")
    cosine, sine = math.cos(swing), math.sin(swing)
    bucket_x_in_swing = cosine * bucket_x - sine * bucket_y
    return math.atan2(-bucket_z, bucket_x_in_swing)


def waypoint_values_to_unity(values: Sequence[float]) -> Tuple[float, ...]:
    if len(values) != 12:
        raise ValueError("waypoint observation slice must contain 12 values")
    converted = []
    for offset in range(0, 9, 3):
        converted.extend(position_to_unity(values[offset : offset + 3]))
    converted.extend(float(value) for value in values[9:])
    return tuple(converted)


def build_observation(
    *,
    machine_profile: Mapping[str, Any],
    machine_state: Mapping[str, Any],
    bucket_tip: BucketTipObservation,
    waypoint_values: Sequence[float],
    previous_action: Sequence[float],
    previous_tip: Optional[BucketTipObservation],
    task_mode: str,
    episode_progress: float = 0.0,
) -> Tuple[float, ...]:
    schema = machine_profile["observation_schema"]
    if int(schema["total_dim"]) != 38:
        raise ValueError("edge runtime only supports the 38D observation contract")
    if len(waypoint_values) != 12:
        raise ValueError("waypoint observation slice must contain 12 values")
    if len(previous_action) != 4:
        raise ValueError("previous action must contain four values")

    actuators = machine_profile["actuators"]
    actuator_state = machine_state["actuator_state"]
    normalizers = schema["normalizers"]
    observation = []

    for name in ("boom", "stick", "bucket"):
        observation.append(
            normalize_position(
                actuator_state[name]["position_m"],
                actuators[name],
            )
        )
        observation.append(
            normalize_velocity(
                actuator_state[name]["velocity_mps"],
                actuators[name],
            )
        )

    swing_angle = float(actuator_state["swing"]["position_rad"]) * observation_sign(
        actuators["swing"]
    )
    observation.extend(
        (
            math.sin(swing_angle),
            math.cos(swing_angle),
            normalize_velocity(
                actuator_state["swing"]["velocity_rad_s"],
                actuators["swing"],
            ),
        )
    )

    position_normalizer = max(
        float(normalizers["position_normalizer"]),
        0.01,
    )
    observation.extend(value / position_normalizer for value in bucket_tip.position_m)
    observation.extend(
        _tip_velocity(
            previous_tip,
            bucket_tip,
            float(normalizers["tip_velocity_scale"]),
        )
    )
    observation.extend(float(value) for value in waypoint_values)
    observation.extend(
        (
            1.0 if task_mode == "MoveToDig" else 0.0,
            1.0 if task_mode == "CarryMaterial" else 0.0,
            clamp(episode_progress, 0.0, 1.0),
        )
    )
    observation.extend(clamp(value) for value in previous_action)

    pitch_deg = math.degrees(bucket_tip.pitch_rad)
    target_pitch_deg = float(
        machine_profile["task_profile"]["bucket_pitch_targets_deg"][task_mode]
    )
    pitch_error_deg = (pitch_deg - target_pitch_deg + 180.0) % 360.0 - 180.0
    pitch_norm_deg = max(float(normalizers["pitch_norm_deg"]), 1.0)
    observation.extend(
        (
            pitch_deg / pitch_norm_deg,
            target_pitch_deg / pitch_norm_deg,
            pitch_error_deg / pitch_norm_deg,
            _pitch_velocity(previous_tip, bucket_tip, pitch_deg),
        )
    )
    if len(observation) != 38 or not all(math.isfinite(value) for value in observation):
        raise ValueError("constructed observation is not 38 finite values")
    return tuple(observation)


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def observation_sign(actuator: Mapping[str, Any]) -> float:
    value = actuator.get("deploy_observation_sign", actuator.get("sign", 1))
    if isinstance(value, bool) or value not in (-1, 1):
        raise ValueError("deploy_observation_sign must be -1 or +1")
    return float(value)


def normalize_position(raw_position: float, actuator: Mapping[str, Any]) -> float:
    deploy = actuator.get("deploy_position_observation")
    configured_range = deploy.get("range") if isinstance(deploy, Mapping) else actuator.get("range")
    if not isinstance(configured_range, (list, tuple)) or len(configured_range) != 2:
        raise ValueError("actuator position observation range is invalid")
    lower, upper = (float(value) for value in configured_range)
    if not all(math.isfinite(value) for value in (lower, upper)) or lower >= upper:
        raise ValueError("actuator position observation range is invalid")
    normalized = (float(raw_position) - lower) / (upper - lower) * 2.0 - 1.0
    return clamp(normalized * observation_sign(actuator))


def normalize_velocity(raw_velocity: float, actuator: Mapping[str, Any]) -> float:
    signed_velocity = float(raw_velocity) * observation_sign(actuator)
    speed = float(
        actuator["max_speed_positive"]
        if signed_velocity >= 0.0
        else actuator["max_speed_negative"]
    )
    if speed <= 1e-9:
        return 0.0
    return clamp(signed_velocity / speed)


def _tip_velocity(
    previous: Optional[BucketTipObservation],
    current: BucketTipObservation,
    velocity_scale: float,
) -> Tuple[float, float, float]:
    if previous is None:
        return 0.0, 0.0, 0.0
    dt_s = max((current.stamp_ms - previous.stamp_ms) / 1000.0, 0.0)
    if dt_s <= 1e-6:
        return 0.0, 0.0, 0.0
    scale = max(float(velocity_scale), 0.001)
    return tuple(
        (current.position_m[index] - previous.position_m[index]) / dt_s / scale
        for index in range(3)
    )  # type: ignore[return-value]


def _pitch_velocity(
    previous: Optional[BucketTipObservation],
    current: BucketTipObservation,
    current_pitch_deg: float,
) -> float:
    if previous is None:
        return 0.0
    dt_s = max((current.stamp_ms - previous.stamp_ms) / 1000.0, 0.0)
    if dt_s <= 1e-6:
        return 0.0
    previous_deg = math.degrees(previous.pitch_rad)
    delta_deg = (current_pitch_deg - previous_deg + 180.0) % 360.0 - 180.0
    return clamp(delta_deg / dt_s / 180.0)
