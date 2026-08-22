"""Local Orin edge inference action source using the existing Action Relay."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from .follow import EdgeFollowRuntime, EdgeFollowStep
from .fixed_actions import FixedActionRuntime, FixedActionRuntimeStep


LOGGER = logging.getLogger("orin_edge_control")


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
    ) -> None:
        if valid_for_ms <= 0:
            raise ValueError("edge action valid_for_ms must be positive")
        if not isinstance(retain_action_authority, bool):
            raise ValueError("retain_action_authority must be boolean")
        self._runtime = runtime
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
        self._consecutive_rejections = 0
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
        try:
            step = self._runtime.step(machine_state, now_s=now_s)
        except Exception as exc:
            self._consecutive_rejections += 1
            loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._stop_or_zero(action_stamp_ms)
            self._append(
                {
                    "schema_version": "orin_edge_control_audit.v1",
                    "mode": "control",
                    "status": "rejected",
                    "source_seq": machine_state.get("seq"),
                    "source_stamp_ms": machine_state.get("stamp_ms"),
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "consecutive_rejections": self._consecutive_rejections,
                    "runtime_monotonic_s": float(now_s),
                    "loop_elapsed_ms": loop_elapsed_ms,
                }
            )
            LOGGER.warning(
                "edge control rejected state seq=%s: %s",
                machine_state.get("seq"),
                exc,
            )
            return None

        self._consecutive_rejections = 0
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
        self._dispatch_action(
            action,
            action_stamp_ms,
            policy_active=result_status == "ACTIVE" and not step.completed,
        )
        self._append(
            {
                "schema_version": "orin_edge_control_audit.v1",
                "mode": "control",
                "status": (
                    result_status
                    if result_status == "TIMEOUT"
                    else ("completed" if step.completed else "active")
                ),
                "source_seq": step.source_seq,
                "source_stamp_ms": step.source_stamp_ms,
                "waypoint_index": step.waypoint_index,
                "waypoint_distance_m": getattr(step, "waypoint_distance_m", None),
                "episode_progress": getattr(step, "episode_progress", None),
                "follow_elapsed_s": getattr(step, "follow_elapsed_s", None),
                "tracking_timeout_s": getattr(step, "tracking_timeout_s", None),
                "waypoint_tolerance_m": getattr(step, "waypoint_tolerance_m", None),
                "inference_ms": getattr(step, "inference_ms", None),
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
            }
        )
        return step

    def close(self, *, action_stamp_ms: int) -> None:
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

    def _append(self, record: Mapping[str, Any]) -> None:
        self._audit_handle.write(json.dumps(record, separators=(",", ":")) + "\n")


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
