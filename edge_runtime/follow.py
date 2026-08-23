"""One high-rate local Follow step: state -> FK -> observation -> ONNX -> velocity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .actions import (
    physical_velocity_from_normalized,
    slew_limited_normalized_action,
)
from .kinematics import UrdfBucketTipKinematics
from .observation import (
    BucketTipObservation,
    bucket_pitch_from_ros_pose,
    build_observation,
    position_to_unity,
    waypoint_values_to_unity,
)
from .trajectory import MissionFollowLimits, TrajectorySnapshot, WaypointTracker
from .trajectory_controller import (
    OnnxRlTrajectoryControllerAdapter,
    TrajectoryController,
)


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
    episode_progress: float = 0.0
    waypoint_distance_m: float = 0.0
    follow_elapsed_s: float = 0.0
    tracking_timeout_s: float = 0.0
    waypoint_tolerance_m: float = 0.0
    inference_ms: Optional[float] = None
    result: str = "ACTIVE"
    commanded_normalized_action: Tuple[
        float,
        float,
        float,
        float,
    ] = (0.0, 0.0, 0.0, 0.0)
    trajectory_controller_backend: str = "unknown"


class EdgeFollowRuntime:
    """Stateful policy runner; it never owns or writes the STM32 serial port."""

    def __init__(
        self,
        *,
        machine_profile: Mapping[str, Any],
        kinematics: UrdfBucketTipKinematics,
        policy: Any = None,
        controller: Optional[TrajectoryController] = None,
        trajectory: Mapping[str, Any],
        mission: Mapping[str, Any],
        action_slew_rate_per_s: Optional[float] = None,
    ) -> None:
        if machine_profile.get("machine_id") != "scale_excavator_v1":
            raise ValueError("unsupported machine profile")
        if kinematics.root_link != "fk_root":
            raise ValueError("deployed URDF root must be fk_root")
        self._machine_profile = machine_profile
        self._kinematics = kinematics
        if (policy is None) == (controller is None):
            raise ValueError("exactly one trajectory controller or legacy policy is required")
        self._controller = (
            controller
            if controller is not None
            else OnnxRlTrajectoryControllerAdapter(policy)
        )
        self._controller.reset()
        if action_slew_rate_per_s is not None:
            rate = float(action_slew_rate_per_s)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(
                    "action_slew_rate_per_s must be finite and positive"
                )
            self._action_slew_rate_per_s: Optional[float] = rate
        else:
            self._action_slew_rate_per_s = None
        self._follow_limits = MissionFollowLimits.from_mapping(mission)
        if self._follow_limits.waypoint_dwell_s != 0.0:
            raise ValueError(
                "nonzero mission limits.waypoint_dwell_s is not supported"
            )
        self._tracker = WaypointTracker(
            TrajectorySnapshot.from_mapping(trajectory),
            waypoint_tolerance_m=self._follow_limits.waypoint_tolerance_m,
        )
        self._previous_tip = None
        self._previous_action = (0.0, 0.0, 0.0, 0.0)
        self._previous_commanded_action = (0.0, 0.0, 0.0, 0.0)
        self._last_source_seq = None
        self._follow_started_monotonic = None
        self._last_monotonic = None
        self._terminal_result = None

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
        if self._last_monotonic is not None and now_s < self._last_monotonic:
            raise ValueError("monotonic time is not increasing")
        if self._follow_started_monotonic is None:
            self._follow_started_monotonic = float(now_s)
        elapsed_s = float(now_s) - self._follow_started_monotonic
        episode_progress = min(
            max(elapsed_s / self._follow_limits.tracking_timeout_s, 0.0),
            1.0,
        )
        if (
            self._terminal_result is None
            and elapsed_s >= self._follow_limits.tracking_timeout_s
        ):
            self._terminal_result = "TIMEOUT"

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
        if self._terminal_result is None:
            self._tracker = self._tracker.advance(bucket_tip_ros)
            if self._tracker.completed:
                self._terminal_result = "COMPLETED"
        waypoint_distance_m = math.dist(
            bucket_tip_ros,
            self._tracker.snapshot.waypoints[self._tracker.current_index],
        )
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
            episode_progress=episode_progress,
        )
        if self._terminal_result is None:
            controller_output = self._controller.compute_action(observation)
            normalized = controller_output.normalized_action
            if self._action_slew_rate_per_s is None:
                commanded_normalized = normalized
            else:
                elapsed_since_command_s = (
                    0.0
                    if self._last_monotonic is None
                    else float(now_s) - self._last_monotonic
                )
                commanded_normalized = slew_limited_normalized_action(
                    normalized,
                    self._previous_commanded_action,
                    elapsed_s=elapsed_since_command_s,
                    max_rate_per_s=self._action_slew_rate_per_s,
                )
            physical = physical_velocity_from_normalized(
                commanded_normalized,
                self._machine_profile,
            )
            inference_ms = controller_output.inference_ms
            result_status = "ACTIVE"
        else:
            normalized = (0.0, 0.0, 0.0, 0.0)
            commanded_normalized = (0.0, 0.0, 0.0, 0.0)
            physical = (0.0, 0.0, 0.0, 0.0)
            inference_ms = None
            result_status = self._terminal_result
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
            episode_progress=episode_progress,
            waypoint_distance_m=waypoint_distance_m,
            follow_elapsed_s=elapsed_s,
            tracking_timeout_s=self._follow_limits.tracking_timeout_s,
            waypoint_tolerance_m=self._follow_limits.waypoint_tolerance_m,
            inference_ms=inference_ms,
            result=result_status,
            commanded_normalized_action=commanded_normalized,
            trajectory_controller_backend=self._controller.descriptor.backend_id,
        )
        self._previous_tip = tip
        if result_status == "ACTIVE":
            # Match Unity training: the 38D previous-action slice records the
            # raw policy action, not the actuator controller's ramped speed.
            self._previous_action = normalized
            self._previous_commanded_action = commanded_normalized
        self._last_source_seq = source_seq
        self._last_monotonic = float(now_s)
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
