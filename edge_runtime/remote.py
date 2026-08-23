"""Remote Follow RPC over length-prefixed TCP JSON.

This facade preserves the public Runtime Interface while private Modules own
Snapshot validation and construction. Physical actions still flow exclusively
through :class:`EdgeControlRunner` and the loopback ``ActionRelay``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from ._remote_behavior_state import (
    ActiveFixedAction as _ActiveFixedAction,
    ActiveFollow as _ActiveFollow,
    SafetySnapshot as _SafetySnapshot,
)
from ._remote_follow import EdgeFollowRuntimeFactory, FollowTrajectorySnapshot
from .follow import EdgeFollowRuntime
from .kinematics import UrdfBucketTipKinematics
from .remote_transport import (
    MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    RemoteBehaviorServer,
    receive_message,
    request_identity as _request_identity,
    send_message,
)
from .remote_validation import positive as _positive
from .trajectory_controller import build_trajectory_controller_builder


# Preserve the historical public import/pickle identity while keeping the
# implementation in a focused private Module.
FollowTrajectorySnapshot.__module__ = __name__
EdgeFollowRuntimeFactory.__module__ = __name__


def __getattr__(name: str) -> Any:
    if name != "OnnxPolicy":
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from .onnx_policy import OnnxPolicy
    return OnnxPolicy


class EdgeBehaviorExecutor:
    """Serialize the one active Orin-local behavior and its state observations."""

    def __init__(
        self,
        *,
        runtime_factory: EdgeFollowRuntimeFactory,
        runner_factory: Callable[[EdgeFollowRuntime], Any],
        fixed_action_factory: Any = None,
        fixed_runner_factory: Optional[Callable[[Any, str], Any]] = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        action_stamp_clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        sender_constructed: bool = False,
        state_timeout_s: float = 0.3,
        resident_rl_authorized: Optional[Callable[[], bool]] = None,
    ) -> None:
        if resident_rl_authorized is not None and not callable(resident_rl_authorized):
            raise ValueError("resident_rl_authorized must be callable")
        self._runtime_factory = runtime_factory
        self._runner_factory = runner_factory
        self._fixed_action_factory = fixed_action_factory
        self._fixed_runner_factory = fixed_runner_factory
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._action_stamp_clock = action_stamp_clock
        self._lock = threading.RLock()
        self._active: Optional[Any] = None
        self._event_sequence = 0
        self._last_request_sequences: dict[str, int] = {}
        self._latest_safety: Optional[_SafetySnapshot] = None
        self._last_action_datagrams = 0
        self._sender_constructed = bool(sender_constructed)
        self._state_timeout_s = _positive("state_timeout_s", state_timeout_s)
        self._resident_rl_authorized = resident_rl_authorized

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    def run_when_idle(self, operation: Callable[[], Any]) -> Any:
        """Atomically exclude new RL behaviors while another owner is claimed."""

        if not callable(operation):
            raise ValueError("operation must be callable")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("an RL behavior is active")
            return operation()

    def start(
        self,
        request: Mapping[str, Any],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        with self._lock:
            identity = _request_identity(request, expected_type="start_follow")
            if set(request) != {
                "schema_version",
                "type",
                "session_id",
                "seq",
                "request_id",
                "trajectory",
            }:
                self._reject(identity, event_sink, "BAD_REQUEST", "start_follow fields are invalid")
                return
            if not self._accept_request_sequence(identity, event_sink):
                return
            if self._active is not None:
                self._reject(identity, event_sink, "BUSY", "another behavior is active")
                return
            gate_reason = self._motion_gate_reason_locked(active=None)
            if gate_reason != "ready":
                self._reject(
                    identity,
                    event_sink,
                    "MOTION_NOT_READY",
                    gate_reason,
                )
                return
            try:
                snapshot = FollowTrajectorySnapshot.from_mapping(
                    request["trajectory"],
                    now_s=self._wall_clock(),
                )
                runtime = self._runtime_factory.create(snapshot)
                runner = self._runner_factory(runtime)
            except Exception as exc:
                self._reject(identity, event_sink, "INVALID_TRAJECTORY", str(exc))
                return
            self._active = _ActiveFollow(
                session_id=identity[0],
                request_id=identity[2],
                request_seq=identity[1],
                snapshot=snapshot,
                runner=runner,
                event_sink=event_sink,
                trajectory_controller_backend=_controller_backend(
                    getattr(
                        self._runtime_factory,
                        "trajectory_controller_backend",
                        "unknown",
                    )
                ),
            )
            self._emit(
                event_sink,
                "accepted",
                session_id=identity[0],
                request_id=identity[2],
                trajectory_id=snapshot.trajectory_id,
                trajectory_controller_backend=(
                    self._active.trajectory_controller_backend
                ),
            )

    def handle(
        self,
        request: Mapping[str, Any],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        request_type = request.get("type") if isinstance(request, Mapping) else None
        if request_type == "start_follow":
            self.start(request, event_sink)
            return
        if request_type == "cancel_follow":
            self.cancel(request, event_sink)
            return
        if request_type == "start_fixed_action":
            self._start_fixed_action(request, event_sink)
            return
        if request_type == "cancel_fixed_action":
            self._cancel_fixed_action(request, event_sink)
            return
        identity = _request_identity(request)
        with self._lock:
            self._reject(
                identity,
                event_sink,
                "BAD_REQUEST",
                "unsupported behavior request type",
            )

    def _start_fixed_action(
        self,
        request: Mapping[str, Any],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        with self._lock:
            identity = _request_identity(
                request,
                expected_type="start_fixed_action",
            )
            if set(request) != {
                "schema_version",
                "type",
                "session_id",
                "seq",
                "request_id",
                "behavior",
            }:
                self._reject(
                    identity,
                    event_sink,
                    "BAD_REQUEST",
                    "start_fixed_action fields are invalid",
                )
                return
            if not self._accept_request_sequence(identity, event_sink):
                return
            if self._active is not None:
                self._reject(
                    identity,
                    event_sink,
                    "BUSY",
                    "another behavior is active",
                )
                return
            gate_reason = self._motion_gate_reason_locked(active=None)
            if gate_reason != "ready":
                self._reject(
                    identity,
                    event_sink,
                    "MOTION_NOT_READY",
                    gate_reason,
                )
                return
            behavior = request.get("behavior")
            if behavior not in {"ExecuteDig", "ExecuteDump"}:
                self._reject(
                    identity,
                    event_sink,
                    "INVALID_FIXED_ACTION",
                    "behavior must be ExecuteDig or ExecuteDump",
                )
                return
            if self._fixed_action_factory is None or self._fixed_runner_factory is None:
                self._reject(
                    identity,
                    event_sink,
                    "FIXED_ACTION_UNAVAILABLE",
                    "Orin fixed action runtime is unavailable",
                )
                return
            try:
                runtime = self._fixed_action_factory.create(behavior)
                runner = self._fixed_runner_factory(runtime, behavior)
            except Exception as exc:
                self._reject(
                    identity,
                    event_sink,
                    "INVALID_FIXED_ACTION",
                    str(exc),
                )
                return
            self._active = _ActiveFixedAction(
                session_id=identity[0],
                request_id=identity[2],
                request_seq=identity[1],
                behavior=behavior,
                runner=runner,
                event_sink=event_sink,
            )
            self._emit(
                event_sink,
                "accepted",
                session_id=identity[0],
                request_id=identity[2],
                behavior=behavior,
            )

    def _cancel_fixed_action(
        self,
        request: Mapping[str, Any],
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self._cancel_behavior(
            request,
            expected_type="cancel_fixed_action",
            expected_active_type=_ActiveFixedAction,
            event_sink=event_sink,
            message="fixed action cancelled by client",
        )

    def cancel(
        self,
        request: Mapping[str, Any],
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self._cancel_behavior(
            request,
            expected_type="cancel_follow",
            expected_active_type=_ActiveFollow,
            event_sink=event_sink,
            message="Follow cancelled by client",
        )

    def _cancel_behavior(
        self,
        request: Mapping[str, Any],
        *,
        expected_type: str,
        expected_active_type: type,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]],
        message: str,
    ) -> None:
        with self._lock:
            identity = _request_identity(request, expected_type=expected_type)
            active = self._active
            sink = event_sink or (
                active.event_sink if active is not None else None
            )
            if sink is None:
                raise ValueError("event_sink is required when no behavior is active")
            if set(request) != {
                "schema_version",
                "type",
                "session_id",
                "seq",
                "request_id",
            }:
                self._reject(
                    identity,
                    sink,
                    "BAD_REQUEST",
                    "%s fields are invalid" % expected_type,
                )
                return
            if not self._accept_request_sequence(identity, sink):
                return
            if active is None:
                self._reject(identity, sink, "NOT_ACTIVE", "no behavior is active")
                return
            if not isinstance(active, expected_active_type):
                self._reject(
                    identity,
                    sink,
                    "BEHAVIOR_MISMATCH",
                    "cancel request does not match the active behavior",
                )
                return
            if event_sink is not None and event_sink is not active.event_sink:
                self._reject(
                    identity,
                    sink,
                    "SESSION_MISMATCH",
                    "cancel connection does not own the active behavior",
                )
                return
            if identity[0] != active.session_id:
                self._reject(identity, sink, "SESSION_MISMATCH", "session_id is not active")
                return
            if identity[1] <= active.request_seq:
                self._reject(identity, sink, "OUT_OF_ORDER", "request seq is not increasing")
                return
            self._finish(
                outcome="CANCELLED",
                reason_code="CANCELLED",
                message=message,
            )

    def observe(self, machine_state: Mapping[str, Any]) -> None:
        with self._lock:
            observed_at = self._monotonic_clock()
            self._record_safety(machine_state, observed_at=observed_at)
            active = self._active
            if active is None:
                return
            resident_gate_reason = self._resident_rl_gate_reason_locked()
            if resident_gate_reason is not None:
                self._finish(
                    outcome="FAILED",
                    reason_code="MOTION_GATE_CLOSED",
                    message=resident_gate_reason,
                )
                return
            try:
                step = active.runner.observe(
                    machine_state,
                    now_s=observed_at,
                    action_stamp_ms=self._action_stamp_clock(),
                )
            except Exception as exc:
                self._finish(
                    outcome="FAILED",
                    reason_code="EXECUTION_ERROR",
                    message=str(exc),
                )
                return
            if step is None:
                self._finish(
                    outcome="FAILED",
                    reason_code="STATE_REJECTED",
                    message="active behavior rejected the machine state",
                )
                return
            if isinstance(active, _ActiveFixedAction):
                self._observe_fixed_action(active, step)
                return
            active = replace(
                active,
                final_waypoint_index=step.waypoint_index,
                final_distance_m=step.waypoint_distance_m,
                trajectory_controller_backend=_controller_backend(
                    getattr(
                        step,
                        "trajectory_controller_backend",
                        active.trajectory_controller_backend,
                    )
                ),
            )
            self._active = active
            result = getattr(
                step,
                "result",
                "COMPLETED" if step.completed else "ACTIVE",
            )
            if result == "ACTIVE" and not step.completed:
                self._emit(
                    active.event_sink,
                    "feedback",
                    session_id=active.session_id,
                    request_id=active.request_id,
                    trajectory_id=active.snapshot.trajectory_id,
                    waypoint_index=step.waypoint_index,
                    waypoint_count=len(active.snapshot.waypoints),
                    distance_m=step.waypoint_distance_m,
                    elapsed_s=step.follow_elapsed_s,
                    bucket_tip_stamp_s=step.source_stamp_ms / 1000.0,
                    bucket_tip=list(step.bucket_tip_ros_m),
                    tracking_state="ACTIVE",
                    trajectory_controller_backend=(
                        active.trajectory_controller_backend
                    ),
                    action_datagrams=_action_datagrams(active.runner),
                )
                return
            if result == "COMPLETED" or step.completed:
                self._finish(
                    outcome="SUCCEEDED",
                    reason_code="SUCCEEDED",
                    message="Follow completed",
                )
            elif result == "TIMEOUT":
                self._finish(
                    outcome="FAILED",
                    reason_code="TRACKING_TIMEOUT",
                    message="Follow tracking timeout",
                )
            else:
                self._finish(
                    outcome="FAILED",
                    reason_code="EXECUTION_ERROR",
                    message="Follow returned invalid terminal state",
                )

    def _observe_fixed_action(
        self,
        active: _ActiveFixedAction,
        step: Any,
    ) -> None:
        self._active = replace(
            active,
            final_step_index=step.step_index,
            final_step_label=step.step_label,
            final_max_error=step.max_error,
        )
        if step.result == "ACTIVE":
            self._emit(
                active.event_sink,
                "feedback",
                session_id=active.session_id,
                request_id=active.request_id,
                behavior=active.behavior,
                step_index=step.step_index,
                step_label=step.step_label,
                phase=step.phase,
                max_error=step.max_error,
                action_datagrams=_action_datagrams(active.runner),
            )
            return
        if step.result == "COMPLETED":
            self._finish(
                outcome="SUCCEEDED",
                reason_code=step.reason_code or "SEQUENCE_COMPLETED",
                message="%s completed" % active.behavior,
            )
        elif step.result == "TIMEOUT":
            self._finish(
                outcome="FAILED",
                reason_code=step.reason_code or "STEP_TIMEOUT",
                message="%s step timeout" % active.behavior,
            )
        else:
            self._finish(
                outcome="FAILED",
                reason_code=step.reason_code or "EXECUTION_ERROR",
                message="%s returned invalid terminal state" % active.behavior,
            )

    def close(self, *, emit_result: bool = True) -> None:
        with self._lock:
            if self._active is not None:
                self._finish(
                    outcome="FAILED",
                    reason_code="CONNECTION_CLOSED",
                    message="remote Follow connection closed",
                    emit_result=emit_result,
                )

    def watchdog(self) -> None:
        """Terminate an active Follow if the local motion gate has closed."""
        with self._lock:
            if self._active is None:
                return
            gate_reason = self._motion_gate_reason_locked(active=None)
            if gate_reason == "ready":
                return
            self._finish(
                outcome="FAILED",
                reason_code="MOTION_GATE_CLOSED",
                message=gate_reason,
            )

    def disconnect(
        self,
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        with self._lock:
            if (
                self._active is not None
                and self._active.event_sink is event_sink
            ):
                self._finish(
                    outcome="FAILED",
                    reason_code="CONNECTION_CLOSED",
                    message="remote Follow connection closed",
                    emit_result=False,
                )

    def status_event(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            safety = self._latest_safety
            fresh = (
                safety is not None
                and self._monotonic_clock() - safety.observed_monotonic_s
                <= self._state_timeout_s
            )
            control_enabled = safety.control_enabled if safety is not None else False
            sensor_valid = safety.sensor_valid if safety is not None else False
            stm32_alive = safety.stm32_alive if safety is not None else False
            estop = safety.estop if safety is not None else False
            fault_free = safety is not None and not safety.fault_flags
            gate_reason = self._motion_gate_reason_locked(active=active)
            return self._next_event(
                "status",
                session_id=active.session_id if active is not None else "server",
                request_id=active.request_id if active is not None else "status",
                state_fresh=fresh,
                control_enabled=control_enabled,
                sensor_valid=sensor_valid,
                stm32_alive=stm32_alive,
                estop=estop,
                fault_free=fault_free,
                quiescent=active is None,
                active_behavior=(
                    "Follow"
                    if isinstance(active, _ActiveFollow)
                    else (
                        active.behavior
                        if isinstance(active, _ActiveFixedAction)
                        else ""
                    )
                ),
                fixed_actions_available=self._fixed_action_factory is not None,
                fixed_actions_validated=bool(
                    self._fixed_action_factory is not None
                    and getattr(
                        getattr(self._fixed_action_factory, "profile", None),
                        "validation_status",
                        "",
                    )
                    == "field_validated"
                ),
                action_datagrams=(
                    _action_datagrams(active.runner)
                    if active is not None
                    else self._last_action_datagrams
                ),
                sender_constructed=self._sender_constructed,
                motion_gate_reason=gate_reason,
            )

    def _motion_gate_reason_locked(
        self,
        *,
        active: Optional[Any],
    ) -> str:
        safety = self._latest_safety
        if safety is None:
            return "state_unavailable"
        if (
            self._monotonic_clock() - safety.observed_monotonic_s
            > self._state_timeout_s
        ):
            return "state_stale"
        if safety.estop:
            return "estop"
        if not safety.stm32_alive:
            return "stm32_timeout"
        if not safety.sensor_valid:
            return "sensor_invalid"
        if safety.fault_flags:
            return "fault_flags"
        if not safety.control_enabled:
            return "control_disabled"
        if not self._sender_constructed:
            return "sender_unavailable"
        resident_gate_reason = self._resident_rl_gate_reason_locked()
        if resident_gate_reason is not None:
            return resident_gate_reason
        if active is not None:
            return "behavior_active"
        return "ready"

    def _resident_rl_gate_reason_locked(self) -> str | None:
        """Require Mission-selected RL authority for resident behavior RPCs.

        Legacy non-resident deployments omit the callback and keep their
        existing local motion gate.  Resident deployments supply the current
        core predicate, so a Follow/fixed-action runner cannot claim RL merely
        because an external behavior request arrived while ACT owns motion.
        """

        authorized = self._resident_rl_authorized
        if authorized is None:
            return None
        try:
            is_authorized = authorized()
        except Exception:
            return "resident_rl_authorization_unavailable"
        if is_authorized is not True:
            return "resident_rl_not_active"
        return None

    def _finish(
        self,
        *,
        outcome: str,
        reason_code: str,
        message: str,
        emit_result: bool = True,
    ) -> None:
        active = self._active
        if active is None:
            return
        quiescent = False
        try:
            active.runner.close(action_stamp_ms=self._action_stamp_clock())
            quiescent = True
        except Exception as exc:
            quiescent = False
            message = "%s; terminal zero failed: %s" % (message, exc)
        finally:
            self._last_action_datagrams = _action_datagrams(active.runner)
            self._active = None
        if emit_result:
            fields = {
                "session_id": active.session_id,
                "request_id": active.request_id,
                "outcome": outcome,
                "reason_code": reason_code,
                "message": message,
                "quiescence_confirmed": quiescent,
                "action_datagrams": _action_datagrams(active.runner),
            }
            if isinstance(active, _ActiveFollow):
                fields.update(
                    {
                        "trajectory_id": active.snapshot.trajectory_id,
                        "final_waypoint_index": active.final_waypoint_index,
                        "final_distance_m": active.final_distance_m,
                        "trajectory_controller_backend": (
                            active.trajectory_controller_backend
                        ),
                    }
                )
            else:
                fields.update(
                    {
                        "behavior": active.behavior,
                        "final_step_index": active.final_step_index,
                        "final_step_label": active.final_step_label,
                        "final_max_error": active.final_max_error,
                    }
                )
            self._emit(active.event_sink, "result", **fields)

    def _reject(
        self,
        identity: tuple[str, int, str],
        event_sink: Callable[[Mapping[str, Any]], None],
        reason_code: str,
        message: str,
    ) -> None:
        self._emit(
            event_sink,
            "rejected",
            session_id=identity[0],
            request_id=identity[2],
            reason_code=reason_code,
            message=message,
        )

    def _accept_request_sequence(
        self,
        identity: tuple[str, int, str],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> bool:
        last = self._last_request_sequences.get(identity[0])
        if last is not None and identity[1] <= last:
            self._reject(
                identity,
                event_sink,
                "OUT_OF_ORDER",
                "request seq is not increasing",
            )
            return False
        self._last_request_sequences = {
            **self._last_request_sequences,
            identity[0]: identity[1],
        }
        return True

    def _emit(
        self,
        event_sink: Callable[[Mapping[str, Any]], None],
        event_type: str,
        **fields: Any,
    ) -> None:
        event_sink(self._next_event(event_type, **fields))

    def _next_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "schema_version": SCHEMA_VERSION,
            "type": event_type,
            "session_id": fields.pop("session_id"),
            "seq": self._event_sequence,
            "request_id": fields.pop("request_id"),
            **fields,
        }
        self._event_sequence += 1
        return event

    def _record_safety(
        self,
        machine_state: Mapping[str, Any],
        *,
        observed_at: float,
    ) -> None:
        safety = machine_state.get("safety")
        if not isinstance(safety, Mapping):
            return
        raw_faults = safety.get("fault_flags")
        faults = (
            tuple(str(value) for value in raw_faults)
            if isinstance(raw_faults, list)
            else ("fault_flags_invalid",)
        )
        self._latest_safety = _SafetySnapshot(
            control_enabled=safety.get("control_enabled") is True,
            sensor_valid=safety.get("sensor_valid") is True,
            stm32_alive=safety.get("stm32_alive") is True,
            estop=safety.get("estop") is True,
            fault_flags=faults,
            observed_monotonic_s=observed_at,
        )


def _action_datagrams(runner: Any) -> int:
    value = getattr(runner, "action_datagrams", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _controller_backend(value: Any) -> str:
    return value if isinstance(value, str) and value.strip() else "unknown"
