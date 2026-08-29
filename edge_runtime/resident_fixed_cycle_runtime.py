"""Orin-local execution Module for one field-validated fixed cycle.

The PC starts or cancels a complete Mission.  This Module advances every
intermediate RL/ACT/fixed-action phase from local behavior results and STM32
acknowledged authority state.  It never opens the serial port and never writes
an actuator command directly.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping

from ._remote_follow import FollowTrajectorySnapshot
from .remote_transport import SCHEMA_VERSION as BEHAVIOR_SCHEMA_VERSION
from .resident_fixed_cycle import (
    FixedCyclePlan,
    FixedCycleSnapshot,
    FixedTrajectoryArtifact,
    FixedTrajectoryTemplate,
    ResidentFixedCycleCoordinator,
)


LOGGER = logging.getLogger("edge_runtime.resident_fixed_cycle")


@dataclass(frozen=True)
class _PendingBehavior:
    kind: str
    target_id: str
    created_monotonic_s: float


class ResidentFixedCycleRuntime:
    """Drive a complete fixed-target cycle through existing resident seams."""

    def __init__(
        self,
        *,
        plan: FixedCyclePlan,
        registry: Mapping[str, FixedTrajectoryTemplate],
        core: Any,
        behavior_executor: Any,
        act_worker_ready: Callable[[], bool],
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        activation_timeout_s: float = 5.0,
        mission_lease_ms: int = 3000,
        terminal_status_grace_s: float = 3.0,
    ) -> None:
        if not isinstance(plan, FixedCyclePlan):
            raise ValueError("plan must be a FixedCyclePlan")
        expected = plan.trajectory_target_ids
        if tuple(registry) != expected or any(
            not isinstance(registry[target], FixedTrajectoryTemplate)
            for target in expected
        ):
            raise ValueError("registry does not exactly match the fixed cycle plan")
        if not callable(act_worker_ready):
            raise ValueError("act_worker_ready must be callable")
        if (
            isinstance(activation_timeout_s, bool)
            or not isinstance(activation_timeout_s, (int, float))
            or not 0.1 <= float(activation_timeout_s) <= 30.0
        ):
            raise ValueError("activation_timeout_s must be within [0.1, 30.0]")
        self._plan = plan
        self._registry = registry
        self._core = core
        self._behavior_executor = behavior_executor
        self._act_worker_ready = act_worker_ready
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._activation_timeout_s = float(activation_timeout_s)
        if (
            isinstance(mission_lease_ms, bool)
            or not isinstance(mission_lease_ms, int)
            or not 500 <= mission_lease_ms <= 5000
        ):
            raise ValueError("mission_lease_ms must be within [500, 5000]")
        self._mission_lease_ms = mission_lease_ms
        if (
            isinstance(terminal_status_grace_s, bool)
            or not isinstance(terminal_status_grace_s, (int, float))
            or not 1.0 <= float(terminal_status_grace_s) <= 10.0
        ):
            raise ValueError("terminal_status_grace_s must be within [1, 10]")
        self._terminal_status_grace_s = float(terminal_status_grace_s)
        self._lock = threading.RLock()
        self._coordinator = ResidentFixedCycleCoordinator(plan=plan, driver=self)
        self._pending: _PendingBehavior | None = None
        self._session_id = ""
        self._request_sequence = 0
        self._active_act_generation: int | None = None
        self._active_follow_target_id = ""
        self._visualization: dict[str, Any] | None = None
        self._act_activation_started_s: float | None = None
        self._act_was_active = False
        self._terminal_disarmed_s: float | None = None

    @property
    def snapshot(self) -> FixedCycleSnapshot:
        with self._lock:
            return self._coordinator.snapshot

    @property
    def visualization_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._visualization is None:
                return None
            return dict(self._visualization)

    @property
    def owner_release_ready(self) -> bool:
        """Keep the read-only terminal receipt available for a bounded grace."""

        with self._lock:
            terminal_disarmed_s = self._terminal_disarmed_s
            if terminal_disarmed_s is None:
                return not self._core.is_operational
            return (
                self._monotonic_clock() - terminal_disarmed_s
                >= self._terminal_status_grace_s
            )

    def start(
        self,
        *,
        run_id: str,
        requested_cycles: int,
        first_dig_point_id: str | None = None,
        dig_group_id: str | None = None,
    ) -> FixedCycleSnapshot:
        with self._lock:
            self._session_id = run_id
            self._request_sequence = 0
            self._pending = None
            self._active_act_generation = None
            self._active_follow_target_id = ""
            self._visualization = None
            self._act_activation_started_s = None
            self._act_was_active = False
            self._terminal_disarmed_s = None
            self._core.renew_mission_lease(lease_ms=self._mission_lease_ms)
            self._coordinator.start(
                run_id=run_id,
                requested_cycles=requested_cycles,
                first_dig_point_id=first_dig_point_id,
                dig_group_id=dig_group_id,
            )
            return self._coordinator.snapshot

    def tick(self) -> FixedCycleSnapshot:
        """Advance only from local acknowledged control and behavior state."""

        with self._lock:
            snapshot = self._coordinator.snapshot
            if snapshot.stage == "IDLE" or snapshot.terminal:
                return snapshot
            if not self._core.is_operational:
                self._fail("RESIDENT_CORE_NOT_OPERATIONAL")
                return self._coordinator.snapshot
            if snapshot.stage in {"ACT_DIG", "ACT_FULL_CYCLE"}:
                self._advance_act()
            self._dispatch_pending_if_ready()
            return self._coordinator.snapshot

    def heartbeat(self) -> FixedCycleSnapshot:
        """Renew only the supervisory PC lease, never a policy stage."""

        with self._lock:
            snapshot = self._coordinator.snapshot
            if snapshot.stage == "IDLE":
                raise RuntimeError("no active resident fixed cycle can be renewed")
            if snapshot.terminal:
                return snapshot
            if not self._core.is_operational:
                raise RuntimeError("resident motion core is not operational")
            self._core.renew_mission_lease(lease_ms=self._mission_lease_ms)
            return snapshot

    def cancel(self) -> None:
        with self._lock:
            self._pending = None
            self._active_follow_target_id = ""
            self._visualization = None
            self._coordinator.cancel()

    # ResidentFixedCycleDriver implementation.
    def start_follow(self, artifact: FixedTrajectoryArtifact) -> None:
        target_id = self._target_for_artifact(artifact)
        self._queue_behavior("follow", target_id)
        if not self._core.rl_is_active:
            self._core.activate_rl()
        self._dispatch_pending_if_ready()

    def activate_act(self, *, max_steps: int) -> None:
        if self._act_worker_ready() is not True:
            raise RuntimeError("resident ACT worker is not ready")

        def claim_act() -> int:
            return self._core.activate_act(max_steps=max_steps)

        self._active_act_generation = self._behavior_executor.run_when_idle(
            claim_act
        )
        self._active_follow_target_id = ""
        self._visualization = None
        self._act_activation_started_s = self._monotonic_clock()
        self._act_was_active = bool(self._core.act_is_active)

    def start_fixed_action(self, behavior: str) -> None:
        if behavior != "ExecuteDump":
            raise ValueError("V3-A fixed cycle only permits ExecuteDump")
        self._queue_behavior("fixed_action", behavior)
        self._active_follow_target_id = ""
        self._visualization = None
        if not self._core.rl_is_active:
            self._core.activate_rl()
        self._dispatch_pending_if_ready()

    def terminal_disarm(self) -> None:
        self._pending = None
        self._active_follow_target_id = ""
        self._visualization = None
        self._core.terminal_disarm()
        self._terminal_disarmed_s = self._monotonic_clock()

    def _advance_act(self) -> None:
        if self._act_worker_ready() is not True:
            self._fail("ACT_WORKER_UNAVAILABLE")
            return
        status = self._core.control_status_snapshot()
        segment = status.act_segment
        if segment.generation != self._active_act_generation:
            self._fail("ACT_GENERATION_MISMATCH")
            return
        if segment.complete:
            self._active_act_generation = None
            self._act_activation_started_s = None
            reason_code = {
                "step_budget": "STEP_BUDGET_REACHED",
                "deadzone_chunk": "DEADZONE_CHUNK_REACHED",
            }.get(segment.completion_reason)
            if reason_code is None:
                self._fail("ACT_COMPLETION_REASON_INVALID")
                return
            self._coordinator.record_child_result(
                child="act",
                outcome="SUCCEEDED",
                reason_code=reason_code,
                quiescence_confirmed=True,
                completed_steps=segment.completed_steps,
            )
            return
        if status.act_is_active:
            self._act_was_active = True
            return
        if self._act_was_active:
            self._fail("ACT_AUTHORITY_LOST")
            return
        started = self._act_activation_started_s
        if started is not None and self._monotonic_clock() - started > self._activation_timeout_s:
            self._fail("ACT_ACTIVATION_TIMEOUT")

    def _queue_behavior(self, kind: str, target_id: str) -> None:
        if self._pending is not None:
            raise RuntimeError("another local behavior activation is pending")
        self._pending = _PendingBehavior(
            kind=kind,
            target_id=target_id,
            created_monotonic_s=self._monotonic_clock(),
        )

    def _dispatch_pending_if_ready(self) -> None:
        pending = self._pending
        if pending is None or self._coordinator.snapshot.terminal:
            return
        if self._monotonic_clock() - pending.created_monotonic_s > self._activation_timeout_s:
            self._fail("RL_ACTIVATION_TIMEOUT")
            return
        if not self._core.rl_is_active or self._behavior_executor.busy:
            return
        self._pending = None
        self._request_sequence += 1
        if pending.kind == "follow":
            request = self._follow_request(
                self._registry[pending.target_id],
                sequence=self._request_sequence,
            )
            self._active_follow_target_id = pending.target_id
        else:
            request = self._fixed_action_request(
                pending.target_id,
                sequence=self._request_sequence,
            )
        self._behavior_executor.handle(request, self._on_behavior_event)

    def _on_behavior_event(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            event_type = event.get("type")
            if event_type == "feedback":
                if not self._active_follow_target_id:
                    return
                try:
                    self._visualization = _visualization_snapshot(
                        event,
                        target_id=self._active_follow_target_id,
                    )
                except (TypeError, ValueError) as exc:
                    self._visualization = None
                    LOGGER.warning(
                        "ignoring invalid read-only trajectory visualization: %s",
                        exc,
                    )
                return
            if event_type in {"accepted", "status"}:
                return
            snapshot = self._coordinator.snapshot
            if snapshot.terminal:
                return
            child = (
                "fixed_action" if snapshot.stage == "EXECUTE_DUMP" else "follow"
            )
            if event_type == "result":
                self._active_follow_target_id = ""
                self._visualization = None
                self._coordinator.record_child_result(
                    child=child,
                    outcome=event.get("outcome", "FAILED"),
                    reason_code=event.get("reason_code", "BEHAVIOR_FAILED"),
                    quiescence_confirmed=event.get("quiescence_confirmed") is True,
                )
                return
            self._coordinator.record_child_result(
                child=child,
                outcome="FAILED",
                reason_code=str(event.get("reason_code", "BEHAVIOR_REJECTED")),
                quiescence_confirmed=True,
            )

    def _follow_request(
        self,
        template: FixedTrajectoryTemplate,
        *,
        sequence: int,
    ) -> dict[str, Any]:
        now = float(self._wall_clock())
        snapshot = FollowTrajectorySnapshot(
            trajectory_id=template.trajectory_id,
            trajectory_sha256="0" * 64,
            frame_id=template.frame_id,
            created_at_s=now,
            mission_id=template.mission_id,
            mission_sha256=template.mission_sha256,
            mission_phase=template.phase,
            task_mode=template.task_mode,
            planning_scope="execution_strict",
            control_stage=template.control_stage,
            workspace_constraint=template.workspace_constraint,
            execution_eligible=True,
            source_bucket_tip_stamp_s=now,
            source_local_map_stamp_s=now,
            inputs_frozen_at_s=now,
            valid_until_s=now + template.tracking_timeout_s + 5.0,
            input_source="live",
            map_source="live_local_map",
            clock_mode="ros_clock",
            waypoints=template.waypoints,
            waypoint_tolerance_m=template.waypoint_tolerance_m,
            waypoint_dwell_s=template.waypoint_dwell_s,
            tracking_timeout_s=template.tracking_timeout_s,
            intermediate_waypoint_tolerance_m=(
                template.intermediate_waypoint_tolerance_m
            ),
        )
        snapshot = replace(snapshot, trajectory_sha256=snapshot.computed_sha256())
        trajectory = asdict(snapshot)
        trajectory["waypoints"] = [list(point) for point in snapshot.waypoints]
        return {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "type": "start_follow",
            "session_id": self._session_id,
            "seq": sequence,
            "request_id": f"{self._session_id}:follow:{sequence}",
            "trajectory": trajectory,
        }

    def _fixed_action_request(self, behavior: str, *, sequence: int) -> dict[str, Any]:
        return {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "type": "start_fixed_action",
            "session_id": self._session_id,
            "seq": sequence,
            "request_id": f"{self._session_id}:fixed:{sequence}",
            "behavior": behavior,
        }

    def _target_for_artifact(self, artifact: FixedTrajectoryArtifact) -> str:
        for target_id, planned in self._plan.trajectories.items():
            if planned == artifact:
                return target_id
        raise ValueError("trajectory artifact is not part of the active plan")

    def _fail(self, reason_code: str) -> None:
        if self._coordinator.snapshot.terminal:
            return
        self._pending = None
        self._active_follow_target_id = ""
        self._visualization = None
        self._coordinator.fail(reason_code=reason_code)


def _visualization_snapshot(
    event: Mapping[str, Any],
    *,
    target_id: str,
) -> dict[str, Any]:
    raw_waypoints = event.get("trajectory_waypoints")
    if not isinstance(raw_waypoints, list) or not 1 <= len(raw_waypoints) <= 12:
        raise ValueError("trajectory visualization waypoints are invalid")
    waypoints: list[tuple[float, float, float]] = []
    for point in raw_waypoints:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("trajectory visualization point is invalid")
        parsed = tuple(float(axis) for axis in point)
        if not all(math.isfinite(axis) for axis in parsed):
            raise ValueError("trajectory visualization point must be finite")
        waypoints.append(parsed)
    current = event.get("waypoint_index")
    if isinstance(current, bool) or not isinstance(current, int):
        raise ValueError("trajectory visualization index is invalid")
    if not 0 <= current < len(waypoints):
        raise ValueError("trajectory visualization index is out of range")
    tolerance = event.get("waypoint_tolerance_m")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("trajectory visualization tolerance is invalid")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or not 0.0 < tolerance <= 5.0:
        raise ValueError("trajectory visualization tolerance is invalid")
    return {
        "frame_id": "machine_root_ros",
        "target_id": target_id,
        "waypoints": tuple(waypoints),
        "current_waypoint_index": current,
        "waypoint_tolerance_m": tolerance,
    }
