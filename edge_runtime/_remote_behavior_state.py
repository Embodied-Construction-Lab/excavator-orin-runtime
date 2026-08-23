"""Immutable state records for one active remote Machine Behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ._remote_follow import FollowTrajectorySnapshot


@dataclass(frozen=True)
class ActiveFollow:
    session_id: str
    request_id: str
    request_seq: int
    snapshot: FollowTrajectorySnapshot
    runner: Any
    event_sink: Callable[[Mapping[str, Any]], None]
    trajectory_controller_backend: str = "unknown"
    final_waypoint_index: int = 0
    final_distance_m: float = -1.0


@dataclass(frozen=True)
class ActiveFixedAction:
    session_id: str
    request_id: str
    request_seq: int
    behavior: str
    runner: Any
    event_sink: Callable[[Mapping[str, Any]], None]
    final_step_index: int = 0
    final_step_label: str = "not_started"
    final_max_error: float = 0.0


@dataclass(frozen=True)
class SafetySnapshot:
    control_enabled: bool
    sensor_valid: bool
    stm32_alive: bool
    estop: bool
    fault_flags: tuple[str, ...]
    observed_monotonic_s: float
