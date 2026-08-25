"""Immutable trajectory snapshot validation and waypoint observation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple


Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class MissionFollowLimits:
    waypoint_tolerance_m: float
    intermediate_waypoint_tolerance_m: float
    waypoint_dwell_s: float
    tracking_timeout_s: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MissionFollowLimits":
        if value.get("schema_version") != "excavation_mission.v1":
            raise ValueError("mission schema_version must be excavation_mission.v1")
        if value.get("frame_id") != "machine_root_ros":
            raise ValueError("mission frame_id must be machine_root_ros")
        limits = value.get("limits")
        if not isinstance(limits, Mapping):
            raise ValueError("mission limits are missing")
        waypoint_tolerance_m = _positive(
            "mission limits.waypoint_tolerance_m",
            limits.get("waypoint_tolerance_m"),
        )
        return cls(
            waypoint_tolerance_m=waypoint_tolerance_m,
            intermediate_waypoint_tolerance_m=_positive(
                "mission limits.intermediate_waypoint_tolerance_m",
                limits.get(
                    "intermediate_waypoint_tolerance_m",
                    waypoint_tolerance_m,
                ),
            ),
            waypoint_dwell_s=_nonnegative(
                "mission limits.waypoint_dwell_s",
                limits.get("waypoint_dwell_s"),
            ),
            tracking_timeout_s=_positive(
                "mission limits.tracking_timeout_s",
                limits.get("tracking_timeout_s"),
            ),
        )


@dataclass(frozen=True)
class TrajectorySnapshot:
    frame_id: str
    task_mode: str
    waypoints: Tuple[Point3, ...]
    target_threshold_m: float
    tube_radius_m: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrajectorySnapshot":
        if value.get("schema_version") != "trajectory_command.v1":
            raise ValueError("trajectory schema_version must be trajectory_command.v1")
        if value.get("frame_id") != "machine_root_ros":
            raise ValueError("trajectory frame_id must be machine_root_ros")
        task_mode = value.get("task_mode")
        if task_mode not in ("MoveToDig", "CarryMaterial"):
            raise ValueError("trajectory task_mode is invalid")
        raw_waypoints = value.get("waypoints_base")
        if not isinstance(raw_waypoints, list) or not raw_waypoints:
            raise ValueError("trajectory must contain waypoints_base")
        waypoints = tuple(_point(point) for point in raw_waypoints)
        if int(value.get("waypoint_count", -1)) != len(waypoints):
            raise ValueError("trajectory waypoint_count does not match waypoints_base")
        threshold = _positive("target_threshold", value.get("target_threshold"))
        tube_radius = _positive("tube_radius", value.get("tube_radius"))
        return cls(
            frame_id="machine_root_ros",
            task_mode=task_mode,
            waypoints=waypoints,
            target_threshold_m=threshold,
            tube_radius_m=tube_radius,
        )


@dataclass(frozen=True)
class WaypointTracker:
    snapshot: TrajectorySnapshot
    waypoint_tolerance_m: float
    intermediate_waypoint_tolerance_m: float | None = None
    current_index: int = 0
    completed: bool = False

    def __post_init__(self) -> None:
        _positive("waypoint_tolerance_m", self.waypoint_tolerance_m)
        if self.intermediate_waypoint_tolerance_m is not None:
            _positive(
                "intermediate_waypoint_tolerance_m",
                self.intermediate_waypoint_tolerance_m,
            )

    @property
    def current_tolerance_m(self) -> float:
        last_index = len(self.snapshot.waypoints) - 1
        if self.current_index in (0, last_index):
            return self.waypoint_tolerance_m
        return (
            self.waypoint_tolerance_m
            if self.intermediate_waypoint_tolerance_m is None
            else self.intermediate_waypoint_tolerance_m
        )

    def advance(self, bucket_tip_ros_m: Sequence[float]) -> "WaypointTracker":
        point = _point(bucket_tip_ros_m)
        if self.completed:
            return self
        target = self.snapshot.waypoints[self.current_index]
        if math.dist(point, target) > self.current_tolerance_m:
            return self
        if self.current_index == len(self.snapshot.waypoints) - 1:
            return replace(self, completed=True)
        return replace(self, current_index=self.current_index + 1)

    def observation_values(
        self,
        *,
        bucket_tip_ros_m: Sequence[float],
        distance_normalizer: float,
        lookahead: int,
    ) -> Tuple[float, ...]:
        tip = _point(bucket_tip_ros_m)
        distance_scale = _positive("distance_normalizer", distance_normalizer)
        if lookahead != 3:
            raise ValueError("38D observation requires waypoint_lookahead=3")
        values = []
        last_index = len(self.snapshot.waypoints) - 1
        for offset in range(lookahead):
            waypoint = self.snapshot.waypoints[
                min(self.current_index + offset, last_index)
            ]
            values.extend(
                (waypoint[index] - tip[index]) / distance_scale
                for index in range(3)
            )
        progress = self.current_index / max(len(self.snapshot.waypoints) - 1, 1)
        tube_ratio = _tube_ratio(
            self.snapshot.waypoints,
            tip,
            self.current_index,
            self.snapshot.tube_radius_m,
        )
        values.extend(
            (
                min(max(progress, 0.0), 1.0),
                tube_ratio,
                1.0 if self.current_index >= last_index else 0.0,
            )
        )
        return tuple(values)


def _tube_ratio(
    waypoints: Sequence[Point3],
    point: Point3,
    current_index: int,
    radius: float,
) -> float:
    if len(waypoints) < 2:
        return 0.0
    start_index = max(current_index - 1, 0)
    end_index = min(current_index, len(waypoints) - 1)
    if start_index == end_index:
        end_index = min(start_index + 1, len(waypoints) - 1)
    distance = _point_to_segment_distance(
        point,
        waypoints[start_index],
        waypoints[end_index],
    )
    return min(max(distance / radius, 0.0), 1.0)


def _point_to_segment_distance(point: Point3, start: Point3, end: Point3) -> float:
    segment = tuple(end[index] - start[index] for index in range(3))
    offset = tuple(point[index] - start[index] for index in range(3))
    length_squared = sum(value * value for value in segment)
    if length_squared <= 1e-12:
        return math.dist(point, start)
    ratio = sum(offset[index] * segment[index] for index in range(3)) / length_squared
    ratio = min(max(ratio, 0.0), 1.0)
    projection = tuple(start[index] + ratio * segment[index] for index in range(3))
    return math.dist(point, projection)


def validate_trajectory_mission(
    trajectory: Mapping[str, Any],
    mission: Mapping[str, Any],
    *,
    mission_sha256: str,
) -> None:
    if trajectory.get("planning_scope") != "execution_strict":
        raise ValueError("trajectory planning_scope must be execution_strict")
    if trajectory.get("execution_eligible") is not True:
        raise ValueError("trajectory execution_eligible must be true")
    provenance = trajectory.get("mission")
    if not isinstance(provenance, Mapping):
        raise ValueError("trajectory mission provenance is missing")
    if provenance.get("id") != mission.get("mission_id"):
        raise ValueError("trajectory mission id does not match mission asset")
    if provenance.get("sha256") != mission_sha256:
        raise ValueError("trajectory mission sha256 does not match mission asset")
    expected_phase = {
        "MoveToDig": "dig",
        "CarryMaterial": "dump",
    }.get(trajectory.get("task_mode"))
    if provenance.get("phase") != expected_phase:
        raise ValueError("trajectory mission phase does not match task_mode")


def _point(values: Sequence[float]) -> Point3:
    if len(values) != 3:
        raise ValueError("point must contain exactly three values")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("point must contain finite values")
    return converted  # type: ignore[return-value]


def _positive(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("%s must be positive and finite" % name)
    return float(value)


def _nonnegative(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError("%s must be nonnegative and finite" % name)
    return float(value)
