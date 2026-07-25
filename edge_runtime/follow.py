"""One high-rate local Follow step: state -> FK -> observation -> ONNX -> velocity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from .actions import physical_velocity_from_normalized
from .kinematics import UrdfBucketTipKinematics
from .observation import (
    BucketTipObservation,
    bucket_pitch_from_ros_pose,
    build_observation,
    position_to_unity,
    waypoint_values_to_unity,
)
from .trajectory import TrajectorySnapshot, WaypointTracker


class Policy(Protocol):
    def run(self, observation: Sequence[float]) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class EdgeFollowStep:
    source_seq: int
    source_stamp_ms: int
    waypoint_index: int
    completed: bool
    bucket_tip_ros_m: Tuple[float, float, float]
    bucket_pitch_rad: float
    observation: Tuple[float, ...]
    normalized_action: Tuple[float, float, float, float]
    physical_action: Tuple[float, float, float, float]


class EdgeFollowRuntime:
    """Stateful policy runner; it never owns or writes the STM32 serial port."""

    def __init__(
        self,
        *,
        machine_profile: Mapping[str, Any],
        kinematics: UrdfBucketTipKinematics,
        policy: Policy,
        trajectory: Mapping[str, Any],
    ) -> None:
        if machine_profile.get("machine_id") != "scale_excavator_v1":
            raise ValueError("unsupported machine profile")
        if kinematics.root_link != "fk_root":
            raise ValueError("deployed URDF root must be fk_root")
        self._machine_profile = machine_profile
        self._kinematics = kinematics
        self._policy = policy
        self._tracker = WaypointTracker(TrajectorySnapshot.from_mapping(trajectory))
        self._previous_tip = None
        self._previous_action = (0.0, 0.0, 0.0, 0.0)
        self._last_source_seq = None

    def step(
        self,
        machine_state: Mapping[str, Any],
        *,
        now_s: float,
    ) -> EdgeFollowStep:
        if not math.isfinite(float(now_s)):
            raise ValueError("now_s must be finite")
        self._validate_state(machine_state)
        source_seq = int(machine_state["seq"])
        if self._last_source_seq is not None and source_seq <= self._last_source_seq:
            raise ValueError("machine_state sequence is not increasing")

        joints = machine_state["joint_state"]["position_rad"]
        pose = self._kinematics.evaluate(
            {
                "swing_joint": joints["swing"],
                "boom_joint": joints["boom"],
                "arm_joint": joints["arm"],
                "bucket_joint": joints["bucket"],
            }
        )
        # The deployed PC launch publishes the explicit identity adapter
        # machine_root_ros -> fk_root. No Unity handedness transform is a TF.
        bucket_tip_ros = pose.position_m
        self._tracker = self._tracker.advance(bucket_tip_ros)
        normalizers = self._machine_profile["observation_schema"]["normalizers"]
        waypoint_ros = self._tracker.observation_values(
            bucket_tip_ros_m=bucket_tip_ros,
            distance_normalizer=float(normalizers["distance_normalizer"]),
            lookahead=int(
                self._machine_profile["observation_schema"]["waypoint_lookahead"]
            ),
        )
        pitch_rad = bucket_pitch_from_ros_pose(
            pose.orientation_xyzw,
            swing_joint_rad=float(joints["swing"]),
        )
        tip = BucketTipObservation(
            position_m=position_to_unity(bucket_tip_ros),
            pitch_rad=pitch_rad,
            stamp_ms=int(machine_state["stamp_ms"]),
        )
        observation = build_observation(
            machine_profile=self._machine_profile,
            machine_state=machine_state,
            bucket_tip=tip,
            waypoint_values=waypoint_values_to_unity(waypoint_ros),
            previous_action=self._previous_action,
            previous_tip=self._previous_tip,
            task_mode=self._tracker.snapshot.task_mode,
        )
        normalized = tuple(float(value) for value in self._policy.run(observation))
        if len(normalized) != 4 or not all(math.isfinite(value) for value in normalized):
            raise ValueError("policy output must contain four finite values")
        physical = physical_velocity_from_normalized(
            normalized,
            self._machine_profile,
        )
        result = EdgeFollowStep(
            source_seq=source_seq,
            source_stamp_ms=int(machine_state["stamp_ms"]),
            waypoint_index=self._tracker.current_index,
            completed=self._tracker.completed,
            bucket_tip_ros_m=bucket_tip_ros,
            bucket_pitch_rad=pitch_rad,
            observation=observation,
            normalized_action=normalized,  # type: ignore[arg-type]
            physical_action=physical,
        )
        self._previous_tip = tip
        self._previous_action = normalized
        self._last_source_seq = source_seq
        return result

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        if state.get("type") != "machine_state_v1":
            raise ValueError("bad machine_state type")
        if state.get("machine_id") != self._machine_profile["machine_id"]:
            raise ValueError("machine_state machine_id does not match profile")
        safety = state.get("safety")
        if not isinstance(safety, Mapping):
            raise ValueError("machine_state safety is missing")
        if safety.get("estop"):
            raise ValueError("estop")
        if not safety.get("stm32_alive"):
            raise ValueError("stm32_not_alive")
        if not safety.get("sensor_valid"):
            raise ValueError("sensor_invalid")
        if safety.get("fault_flags"):
            raise ValueError("fault_flags")
