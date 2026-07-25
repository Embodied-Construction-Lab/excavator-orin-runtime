"""Local Orin edge inference action source using the existing Action Relay."""

from __future__ import annotations

import json
import logging
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

    def observe(
        self,
        machine_state: Mapping[str, Any],
        *,
        now_s: float,
        action_stamp_ms: int,
    ) -> Optional[EdgeFollowStep]:
        try:
            step = self._runtime.step(machine_state, now_s=now_s)
        except Exception as exc:
            self._send((0.0, 0.0, 0.0, 0.0), action_stamp_ms)
            self._append(
                {
                    "schema_version": "orin_edge_control_audit.v1",
                    "mode": "control",
                    "status": "rejected",
                    "source_seq": machine_state.get("seq"),
                    "source_stamp_ms": machine_state.get("stamp_ms"),
                    "reason": str(exc),
                }
            )
            LOGGER.warning(
                "edge control rejected state seq=%s: %s",
                machine_state.get("seq"),
                exc,
            )
            return None

        action = (
            (0.0, 0.0, 0.0, 0.0)
            if step.completed
            else step.physical_action
        )
        self._send(action, action_stamp_ms)
        self._append(
            {
                "schema_version": "orin_edge_control_audit.v1",
                "mode": "control",
                "status": "completed" if step.completed else "active",
                "source_seq": step.source_seq,
                "source_stamp_ms": step.source_stamp_ms,
                "waypoint_index": step.waypoint_index,
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
