"""Local Orin edge inference action source using the existing Action Relay."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from .follow import EdgeFollowRuntime, EdgeFollowStep


LOGGER = logging.getLogger("orin_edge_control")


class ActionSink(Protocol):
    def send(self, payload: bytes) -> object:
        ...


class EdgeControlRunner:
    """Convert each local inference result into one loopback policy_action."""

    def __init__(
        self,
        *,
        runtime: EdgeFollowRuntime,
        action_sink: ActionSink,
        audit_path: Path,
        valid_for_ms: int,
    ) -> None:
        if valid_for_ms <= 0:
            raise ValueError("edge action valid_for_ms must be positive")
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
        self._action_seq = 0
        self._consecutive_rejections = 0

    @property
    def action_datagrams(self) -> int:
        return self._action_seq

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
            self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)
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
        self._send(action, action_stamp_ms)
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
                "physical_action": list(action),
            }
        )
        return step

    def close(self, *, action_stamp_ms: int) -> None:
        try:
            self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)
        finally:
            self._audit_handle.close()

    def _send(self, action: tuple, action_stamp_ms: int) -> None:
        packet = {
            "type": "policy_action",
            "schema_version": "1.0",
            "seq": self._action_seq,
            "stamp_ms": int(action_stamp_ms),
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": [float(value) for value in action],
            # Compatibility debt: values are physical velocities despite this name.
            "action_type": "normalized_velocity_command",
            "valid_for_ms": self._valid_for_ms,
        }
        payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self._action_sink.send(payload)
        self._action_seq += 1

    def _append(self, record: Mapping[str, Any]) -> None:
        self._audit_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
