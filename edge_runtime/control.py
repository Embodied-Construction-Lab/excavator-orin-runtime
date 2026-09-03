"""Local Orin edge inference action source using the existing Action Relay."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from .audit_writer import _BoundedJsonlAuditWriter
from .follow import EdgeFollowRuntime, EdgeFollowStep
from .fixed_actions import FixedActionRuntime, FixedActionRuntimeStep


LOGGER = logging.getLogger("orin_edge_control")


def _tracking_reference(step: Any) -> Optional[tuple[float, float, float]]:
    raw = getattr(step, "reference_waypoint_ros_m", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw
    ):
        return None
    return tuple(float(value) for value in raw)  # type: ignore[return-value]


def _tracking_point(step: Any, field: str) -> tuple[float, float, float]:
    raw = getattr(step, field, None)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{field} must contain three finite values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw
    ):
        raise ValueError(f"{field} must contain three finite values")
    return tuple(float(value) for value in raw)  # type: ignore[return-value]


def _tracking_scalar(
    step: Any,
    field: str,
    *,
    maximum: Optional[float] = None,
) -> float:
    raw = getattr(step, field, None)
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) < 0.0
        or (maximum is not None and float(raw) > maximum)
    ):
        suffix = f" within [0, {maximum:g}]" if maximum is not None else " >= 0"
        raise ValueError(f"{field} must be finite and{suffix}")
    return float(raw)


def _active_tracking_context(step: Any) -> dict[str, Any]:
    reference_waypoint = _tracking_reference(step)
    if reference_waypoint is None:
        raise ValueError(
            "reference_waypoint_ros_m must contain three finite values"
        )
    waypoint_index = getattr(step, "waypoint_index", None)
    if (
        isinstance(waypoint_index, bool)
        or not isinstance(waypoint_index, int)
        or waypoint_index < 0
    ):
        raise ValueError("waypoint_index must be a non-negative integer")
    return {
        "source_seq": getattr(step, "source_seq", None),
        "source_stamp_ms": getattr(step, "source_stamp_ms", None),
        "waypoint_index": waypoint_index,
        "waypoint_distance_m": _tracking_scalar(step, "waypoint_distance_m"),
        "episode_progress": _tracking_scalar(
            step,
            "episode_progress",
            maximum=1.0,
        ),
        "tracking_timeout_s": getattr(step, "tracking_timeout_s", None),
        "waypoint_tolerance_m": getattr(step, "waypoint_tolerance_m", None),
        "bucket_tip_ros_m": list(_tracking_point(step, "bucket_tip_ros_m")),
        "reference_waypoint_ros_m": list(reference_waypoint),
    }


def _runtime_trajectory_controller_backend(runtime: Any) -> str:
    direct = getattr(runtime, "trajectory_controller_backend", None)
    if isinstance(direct, str) and direct.strip():
        return direct
    # EdgeControlRunner and EdgeFollowRuntime are one package-owned boundary;
    # inspect the configured controller before the first step so a rejected
    # first state still records which experiment backend was active.
    controller = getattr(runtime, "_controller", None)
    descriptor = getattr(controller, "descriptor", None)
    backend = getattr(descriptor, "backend_id", None)
    if isinstance(backend, str) and backend.strip():
        return backend
    return "unknown"


class ActionSink(Protocol):
    def send(self, payload: bytes) -> object:
        ...


class ActionSequence:
    """Process-scoped monotonic sequence shared by consecutive Follow runners."""

    def __init__(self, start: int = 0) -> None:
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("action sequence start must be a nonnegative integer")
        self._next_value = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            value = self._next_value
            self._next_value += 1
            return value


class EdgeControlRunner:
    """Convert each local inference result into one loopback policy_action."""

    def __init__(
        self,
        *,
        runtime: EdgeFollowRuntime,
        action_sink: ActionSink,
        audit_path: Path,
        valid_for_ms: int,
        action_sequence: Optional[ActionSequence] = None,
        retain_action_authority: bool = False,
        trace_run_id: Optional[str] = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if valid_for_ms <= 0:
            raise ValueError("edge action valid_for_ms must be positive")
        if not isinstance(retain_action_authority, bool):
            raise ValueError("retain_action_authority must be boolean")
        if trace_run_id is not None and (
            not isinstance(trace_run_id, str) or not trace_run_id.strip()
        ):
            raise ValueError("trace_run_id must be a non-empty string")
        if not callable(monotonic_clock):
            raise ValueError("monotonic_clock must be callable")
        self._runtime = runtime
        self._action_sink = action_sink
        self._audit_path = Path(audit_path)
        self._audit_writer: Optional[_BoundedJsonlAuditWriter] = None
        self._audit_enqueue_disabled = False
        try:
            self._audit_writer = _BoundedJsonlAuditWriter(
                self._audit_path,
            )
        except Exception as exc:
            LOGGER.error(
                "edge control audit disabled while starting writer for %s: %s",
                self._audit_path,
                exc,
            )
        self._valid_for_ms = int(valid_for_ms)
        self._action_sequence = action_sequence or ActionSequence()
        self._action_datagrams = 0
        self._consecutive_rejections = 0
        self._trajectory_controller_backend = (
            _runtime_trajectory_controller_backend(runtime)
        )
        self._trace_run_id = trace_run_id or uuid.uuid4().hex
        self._next_sample_id = 0
        self._trace_started_monotonic_s: Optional[float] = None
        self._trace_terminal_emitted = False
        self._trace_terminal_result: Optional[str] = None
        self._accepted_policy_sample_count = 0
        self._dropped_policy_sample_count = 0
        self._last_tracking_context: dict[str, Any] = {}
        self._monotonic_clock = monotonic_clock
        self._resident_activation_started = False
        self._retain_action_authority = retain_action_authority

    @property
    def action_datagrams(self) -> int:
        return self._action_datagrams

    def observe(
        self,
        machine_state: Mapping[str, Any],
        *,
        now_s: float,
        action_stamp_ms: int,
    ) -> Optional[EdgeFollowStep]:
        started = time.perf_counter()
        if self._trace_started_monotonic_s is None:
            self._trace_started_monotonic_s = float(now_s)
        try:
            step = self._runtime.step(machine_state, now_s=now_s)
        except Exception as exc:
            self._consecutive_rejections += 1
            loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._stop_or_zero(action_stamp_ms)
            self._emit_terminal(
                result="rejected",
                runtime_monotonic_s=float(now_s),
                elapsed_s=self._trace_elapsed_s(float(now_s)),
                extra={
                    "source_seq": machine_state.get("seq"),
                    "source_stamp_ms": machine_state.get("stamp_ms"),
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "consecutive_rejections": self._consecutive_rejections,
                    "loop_elapsed_ms": loop_elapsed_ms,
                },
            )
            LOGGER.warning(
                "edge control rejected state seq=%s: %s",
                machine_state.get("seq"),
                exc,
            )
            return None

        self._consecutive_rejections = 0
        step_backend = getattr(
            step,
            "trajectory_controller_backend",
            self._trajectory_controller_backend,
        )
        if isinstance(step_backend, str) and step_backend.strip():
            self._trajectory_controller_backend = step_backend
        loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
        result_status = getattr(
            step,
            "result",
            "COMPLETED" if step.completed else "ACTIVE",
        )
        action = (
            step.physical_action
            if result_status == "ACTIVE" and not step.completed
            else (0.0, 0.0, 0.0, 0.0)
        )
        policy_active = result_status == "ACTIVE" and not step.completed
        if (
            policy_active
            and self._trace_terminal_emitted
            and self._trace_terminal_result == "rejected"
        ):
            self._trace_run_id = uuid.uuid4().hex
            self._next_sample_id = 0
            self._trace_started_monotonic_s = float(now_s)
            self._trace_terminal_emitted = False
            self._trace_terminal_result = None
            self._accepted_policy_sample_count = 0
            self._dropped_policy_sample_count = 0
            self._last_tracking_context = {}
        if policy_active:
            try:
                tracking_context = _active_tracking_context(step)
            except ValueError as exc:
                self._consecutive_rejections += 1
                self._stop_or_zero(action_stamp_ms)
                self._emit_terminal(
                    result="rejected",
                    runtime_monotonic_s=float(now_s),
                    elapsed_s=self._trace_elapsed_s(float(now_s)),
                    extra={
                        "source_seq": getattr(step, "source_seq", None),
                        "source_stamp_ms": getattr(step, "source_stamp_ms", None),
                        "reason": f"active tracking context invalid: {exc}",
                        "exception_type": type(exc).__name__,
                        "consecutive_rejections": self._consecutive_rejections,
                        "loop_elapsed_ms": loop_elapsed_ms,
                    },
                )
                LOGGER.warning(
                    "edge control rejected active tracking context seq=%s: %s",
                    getattr(step, "source_seq", None),
                    exc,
                )
                return None
        else:
            reference_waypoint = _tracking_reference(step)
            tracking_context = {
                "source_seq": step.source_seq,
                "source_stamp_ms": step.source_stamp_ms,
                "waypoint_index": step.waypoint_index,
                "waypoint_distance_m": getattr(step, "waypoint_distance_m", None),
                "episode_progress": getattr(step, "episode_progress", None),
                "tracking_timeout_s": getattr(step, "tracking_timeout_s", None),
                "waypoint_tolerance_m": getattr(step, "waypoint_tolerance_m", None),
                "bucket_tip_ros_m": list(step.bucket_tip_ros_m),
            }
            if reference_waypoint is not None:
                tracking_context["reference_waypoint_ros_m"] = list(
                    reference_waypoint
                )
        policy_action_seq = self._dispatch_action(
            action,
            action_stamp_ms,
            policy_active=policy_active,
        )
        trace_fields: Mapping[str, Any] = {}
        if policy_active:
            trace_fields = {
                "record_type": "policy_sample",
                "sample_id": self._next_sample_id,
                "trace_run_id": self._trace_run_id,
                "policy_action_seq": policy_action_seq,
                "trace_semantics": "commanded_normalized_action",
                "action_order": ["boom", "stick", "bucket", "swing"],
                "reference_waypoint_ros_m": tracking_context[
                    "reference_waypoint_ros_m"
                ],
            }
            self._next_sample_id += 1
        self._last_tracking_context = tracking_context
        if not policy_active:
            terminal_result = (
                "completed"
                if result_status == "COMPLETED" or step.completed
                else "timeout" if result_status == "TIMEOUT" else "rejected"
            )
            self._emit_terminal(
                result=terminal_result,
                runtime_monotonic_s=float(now_s),
                elapsed_s=float(getattr(step, "follow_elapsed_s", 0.0)),
                extra={
                    **tracking_context,
                    "loop_elapsed_ms": loop_elapsed_ms,
                },
            )
            return step
        accepted = self._append(
            {
                "schema_version": "orin_edge_control_audit.v1",
                "mode": "control",
                "status": "active",
                "result": "active",
                "source_seq": step.source_seq,
                "source_stamp_ms": step.source_stamp_ms,
                "waypoint_index": step.waypoint_index,
                "waypoint_distance_m": getattr(step, "waypoint_distance_m", None),
                "episode_progress": getattr(step, "episode_progress", None),
                "follow_elapsed_s": getattr(step, "follow_elapsed_s", None),
                "tracking_timeout_s": getattr(step, "tracking_timeout_s", None),
                "waypoint_tolerance_m": getattr(step, "waypoint_tolerance_m", None),
                "inference_ms": getattr(step, "inference_ms", None),
                "trajectory_controller_backend": self._trajectory_controller_backend,
                "consecutive_rejections": self._consecutive_rejections,
                "runtime_monotonic_s": float(now_s),
                "loop_elapsed_ms": loop_elapsed_ms,
                "bucket_tip_ros_m": list(step.bucket_tip_ros_m),
                "normalized_action": list(step.normalized_action),
                "commanded_normalized_action": list(
                    getattr(
                        step,
                        "commanded_normalized_action",
                        step.normalized_action,
                    )
                ),
                "physical_action": list(action),
                **trace_fields,
            }
        )
        if accepted:
            self._accepted_policy_sample_count += 1
        else:
            self._dropped_policy_sample_count += 1
        return step

    def close(self, *, action_stamp_ms: int) -> None:
        try:
            self._stop_or_zero(action_stamp_ms)
        finally:
            now_s = float(self._monotonic_clock())
            self._emit_terminal(
                result="interrupted",
                runtime_monotonic_s=now_s,
                elapsed_s=self._trace_elapsed_s(now_s),
                extra=self._last_tracking_context,
            )
            self._close_audit()

    def _trace_elapsed_s(self, now_s: float) -> float:
        if self._trace_started_monotonic_s is None:
            return 0.0
        return max(float(now_s) - self._trace_started_monotonic_s, 0.0)

    def _emit_terminal(
        self,
        *,
        result: str,
        runtime_monotonic_s: float,
        elapsed_s: float,
        extra: Mapping[str, Any],
    ) -> None:
        if self._trace_terminal_emitted:
            return
        accepted = self._append(
            {
                "schema_version": "orin_edge_control_audit.v1",
                "mode": "control",
                "record_type": "terminal",
                "status": "terminal",
                "result": result,
                "trace_run_id": self._trace_run_id,
                "trace_semantics": "commanded_normalized_action",
                "runtime_monotonic_s": float(runtime_monotonic_s),
                "elapsed_s": max(float(elapsed_s), 0.0),
                "trajectory_controller_backend": (
                    self._trajectory_controller_backend
                ),
                **dict(extra),
                "expected_policy_sample_count": self._next_sample_id,
                "accepted_policy_sample_count": (
                    self._accepted_policy_sample_count
                ),
                "dropped_policy_sample_count": self._dropped_policy_sample_count,
            },
            terminal=True,
        )
        if accepted:
            self._trace_terminal_emitted = True
            self._trace_terminal_result = result

    def _dispatch_action(
        self,
        action: tuple[float, float, float, float],
        action_stamp_ms: int,
        *,
        policy_active: bool,
    ) -> Optional[int]:
        if policy_active:
            self._ensure_resident_activation_if_supported()
            return self._send(action, action_stamp_ms)
        return self._stop_or_zero(action_stamp_ms)

    def _stop_or_zero(self, action_stamp_ms: int) -> Optional[int]:
        if self._retain_action_authority:
            if self._resident_activation_started:
                return self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)
            return None
        if self._stop_if_resident_supported():
            return None
        return self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)

    def _ensure_resident_activation_if_supported(self) -> None:
        if self._resident_activation_started:
            return
        begin_activation = getattr(self._action_sink, "begin_activation", None)
        if begin_activation is None:
            return
        begin_activation(now_monotonic_ns=time.monotonic_ns())
        self._resident_activation_started = True

    def _stop_if_resident_supported(self) -> bool:
        request_stop = getattr(self._action_sink, "request_stop", None)
        if request_stop is None:
            return False
        request_stop(now_monotonic_ns=time.monotonic_ns())
        self._resident_activation_started = False
        return True

    def _send(self, action: tuple, action_stamp_ms: int) -> int:
        sequence = self._action_sequence.next()
        packet = {
            "type": "policy_action",
            "schema_version": "1.0",
            "seq": sequence,
            "stamp_ms": int(action_stamp_ms),
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": [float(value) for value in action],
            # Compatibility debt: values are physical velocities despite this name.
            "action_type": "normalized_velocity_command",
            "valid_for_ms": self._valid_for_ms,
        }
        payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self._action_sink.send(payload)
        self._action_datagrams += 1
        return sequence

    def _append(
        self,
        record: Mapping[str, Any],
        *,
        terminal: bool = False,
    ) -> bool:
        writer = self._audit_writer
        if writer is None or (self._audit_enqueue_disabled and not terminal):
            return False
        try:
            return bool(writer.append(record))
        except Exception as exc:
            self._audit_enqueue_disabled = True
            LOGGER.error(
                "edge control audit enqueue disabled at %s: %s",
                self._audit_path,
                exc,
            )
            return False

    def _close_audit(self) -> None:
        writer = self._audit_writer
        self._audit_writer = None
        if writer is None:
            return
        try:
            writer.close()
        except Exception as exc:
            LOGGER.error(
                "edge control audit writer close failed at %s: %s",
                self._audit_path,
                exc,
            )


class FixedActionControlRunner:
    """Send one Orin-local fixed-action control step through the Action Relay."""

    def __init__(
        self,
        *,
        runtime: FixedActionRuntime,
        behavior: str,
        action_sink: ActionSink,
        audit_path: Path,
        valid_for_ms: int,
        action_sequence: Optional[ActionSequence] = None,
        retain_action_authority: bool = False,
    ) -> None:
        if behavior not in {"ExecuteDig", "ExecuteDump"}:
            raise ValueError("fixed action behavior is invalid")
        if valid_for_ms <= 0:
            raise ValueError("edge action valid_for_ms must be positive")
        if not isinstance(retain_action_authority, bool):
            raise ValueError("retain_action_authority must be boolean")
        self._runtime = runtime
        self._behavior = behavior
        self._action_sink = action_sink
        self._audit_path = Path(audit_path)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_handle = self._audit_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._valid_for_ms = int(valid_for_ms)
        self._action_sequence = action_sequence or ActionSequence()
        self._action_datagrams = 0
        self._closed = False
        self._resident_activation_started = False
        self._retain_action_authority = retain_action_authority

    @property
    def action_datagrams(self) -> int:
        return self._action_datagrams

    def observe(
        self,
        machine_state: Mapping[str, Any],
        *,
        now_s: float,
        action_stamp_ms: int,
    ) -> Optional[FixedActionRuntimeStep]:
        started = time.perf_counter()
        try:
            step = self._runtime.step(machine_state, now_s=now_s)
        except Exception as exc:
            self._stop_or_zero(action_stamp_ms)
            self._append(
                {
                    "schema_version": "orin_fixed_action_control_audit.v1",
                    "mode": "control",
                    "behavior": self._behavior,
                    "status": "rejected",
                    "source_seq": machine_state.get("seq"),
                    "source_stamp_ms": machine_state.get("stamp_ms"),
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "runtime_monotonic_s": float(now_s),
                    "loop_elapsed_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            LOGGER.warning(
                "%s rejected state seq=%s: %s",
                self._behavior,
                machine_state.get("seq"),
                exc,
            )
            return None
        action = step.physical_action if step.result == "ACTIVE" else (0.0, 0.0, 0.0, 0.0)
        self._dispatch_action(
            action,
            action_stamp_ms,
            policy_active=step.result == "ACTIVE",
        )
        self._append(
            {
                "schema_version": "orin_fixed_action_control_audit.v1",
                "mode": "control",
                "behavior": self._behavior,
                "status": step.result,
                "reason_code": step.reason_code,
                "source_seq": machine_state.get("seq"),
                "source_stamp_ms": machine_state.get("stamp_ms"),
                "step_index": step.step_index,
                "step_label": step.step_label,
                "phase": step.phase,
                "max_error": step.max_error,
                "normalized_action": list(step.normalized_action),
                "physical_action": list(action),
                "runtime_monotonic_s": float(now_s),
                "loop_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return step

    def close(self, *, action_stamp_ms: int) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stop_or_zero(action_stamp_ms)
        finally:
            self._audit_handle.close()

    def _dispatch_action(
        self,
        action: tuple[float, float, float, float],
        action_stamp_ms: int,
        *,
        policy_active: bool,
    ) -> None:
        if policy_active:
            self._ensure_resident_activation_if_supported()
            self._send(action, action_stamp_ms)
            return
        self._stop_or_zero(action_stamp_ms)

    def _stop_or_zero(self, action_stamp_ms: int) -> None:
        if self._retain_action_authority:
            if self._resident_activation_started:
                self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)
            return
        if self._stop_if_resident_supported():
            return
        self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)

    def _ensure_resident_activation_if_supported(self) -> None:
        if self._resident_activation_started:
            return
        begin_activation = getattr(self._action_sink, "begin_activation", None)
        if begin_activation is None:
            return
        begin_activation(now_monotonic_ns=time.monotonic_ns())
        self._resident_activation_started = True

    def _stop_if_resident_supported(self) -> bool:
        request_stop = getattr(self._action_sink, "request_stop", None)
        if request_stop is None:
            return False
        request_stop(now_monotonic_ns=time.monotonic_ns())
        self._resident_activation_started = False
        return True

    def _send(self, action: tuple, action_stamp_ms: int) -> None:
        packet = {
            "type": "policy_action",
            "schema_version": "1.0",
            "seq": self._action_sequence.next(),
            "stamp_ms": int(action_stamp_ms),
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": [float(value) for value in action],
            "action_type": "normalized_velocity_command",
            "valid_for_ms": self._valid_for_ms,
        }
        self._action_sink.send(
            json.dumps(packet, separators=(",", ":")).encode("utf-8")
        )
        self._action_datagrams += 1

    def _append(self, record: Mapping[str, Any]) -> None:
        self._audit_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
