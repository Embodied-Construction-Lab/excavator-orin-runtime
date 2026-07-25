"""Immutable trajectory snapshot validation and waypoint observation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, Tuple


Point3 = Tuple[float, float, float]


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
    current_index: int = 0
    completed: bool = False

    def advance(self, bucket_tip_ros_m: Sequence[float]) -> "WaypointTracker":
        point = _point(bucket_tip_ros_m)
        if self.completed:
            return self
        target = self.snapshot.waypoints[self.current_index]
        if math.dist(point, target) > self.snapshot.target_threshold_m:
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
