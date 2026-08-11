"""Transport-independent Orin excavation-cycle state machine."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .remote_transport import SCHEMA_VERSION, request_identity


@dataclass(frozen=True)
class CycleDirective:
    """One bounded Machine Behavior requested by the cycle state machine."""

    stage: str
    behavior: str
    trajectory: dict[str, Any] | None = None


@dataclass(frozen=True)
class CycleStatus:
    cycle_id: str = ""
    stage: str = "IDLE"
    active_behavior: str = ""
    terminal: bool = False
    outcome: str = ""
    reason_code: str = ""


class EdgeExcavationCycle:
    """Advance one excavation cycle from explicit child behavior results."""

    def __init__(self) -> None:
        self._status = CycleStatus()

    @property
    def status(self) -> CycleStatus:
        return self._status

    def start(
        self,
        *,
        cycle_id: str,
        dig_trajectory: Mapping[str, Any],
    ) -> CycleDirective:
        if self._status.stage != "IDLE" and not self._status.terminal:
            raise RuntimeError("an excavation cycle is already active")
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ValueError("cycle_id must be non-empty text")
        if not isinstance(dig_trajectory, Mapping):
            raise ValueError("dig_trajectory must be a mapping")
        trajectory = copy.deepcopy(dict(dig_trajectory))
        self._status = CycleStatus(
            cycle_id=cycle_id,
            stage="FOLLOW_DIG",
            active_behavior="Follow",
        )
        return CycleDirective(
            stage="FOLLOW_DIG",
            behavior="Follow",
            trajectory=trajectory,
        )

    def record_behavior_result(
        self,
        *,
        outcome: str,
        reason_code: str,
        quiescence_confirmed: bool,
    ) -> CycleDirective | None:
        if not quiescence_confirmed:
            self._fail("CHILD_NOT_QUIESCENT")
            return None
        if outcome == "CANCELLED":
            self.cancel(reason_code=reason_code or "CANCELLED")
            return None
        if outcome != "SUCCEEDED":
            self._fail(reason_code or "CHILD_FAILED")
            return None

        if self._status.stage == "FOLLOW_DIG" and reason_code == "SUCCEEDED":
            self._status = replace(
                self._status,
                stage="EXECUTE_DIG",
                active_behavior="ExecuteDig",
            )
            return CycleDirective(stage="EXECUTE_DIG", behavior="ExecuteDig")
        if (
            self._status.stage == "EXECUTE_DIG"
            and reason_code == "SEQUENCE_COMPLETED"
        ):
            self._status = replace(
                self._status,
                stage="WAITING_FOR_DUMP_TRAJECTORY",
                active_behavior="",
            )
            return None

        if self._status.stage == "FOLLOW_DUMP" and reason_code == "SUCCEEDED":
            self._status = replace(
                self._status,
                stage="EXECUTE_DUMP",
                active_behavior="ExecuteDump",
            )
            return CycleDirective(stage="EXECUTE_DUMP", behavior="ExecuteDump")
        if (
            self._status.stage == "EXECUTE_DUMP"
            and reason_code == "SEQUENCE_COMPLETED"
        ):
            self._status = replace(
                self._status,
                stage="COMPLETED",
                active_behavior="",
                terminal=True,
                outcome="SUCCEEDED",
                reason_code="SEQUENCE_COMPLETED",
            )
            return None

        self._fail(reason_code or "UNEXPECTED_CHILD_RESULT")
        return None

    def provide_dump_trajectory(
        self,
        dump_trajectory: Mapping[str, Any],
    ) -> CycleDirective:
        if self._status.stage != "WAITING_FOR_DUMP_TRAJECTORY":
            raise RuntimeError("cycle is not waiting for a dump trajectory")
        if not isinstance(dump_trajectory, Mapping):
            raise ValueError("dump_trajectory must be a mapping")
        trajectory = copy.deepcopy(dict(dump_trajectory))
        self._status = replace(
            self._status,
            stage="FOLLOW_DUMP",
            active_behavior="Follow",
        )
        return CycleDirective(
            stage="FOLLOW_DUMP",
            behavior="Follow",
            trajectory=trajectory,
        )

    def cancel(self, *, reason_code: str = "CANCELLED") -> None:
        if self._status.stage == "IDLE" or self._status.terminal:
            raise RuntimeError("no active excavation cycle can be cancelled")
        self._status = replace(
            self._status,
            stage="CANCELLED",
            active_behavior="",
            terminal=True,
            outcome="CANCELLED",
            reason_code=reason_code,
        )

    def _fail(self, reason_code: str) -> None:
        self._status = replace(
            self._status,
            stage="FAILED",
            active_behavior="",
            terminal=True,
            outcome="FAILED",
            reason_code=reason_code,
        )


class EdgeCycleCoordinator:
    """Drive the existing behavior executor from one local cycle state machine."""

    def __init__(self, behavior_executor: Any) -> None:
        if behavior_executor is None:
            raise ValueError("behavior_executor is required")
        self._behavior_executor = behavior_executor
        self._cycle = EdgeExcavationCycle()
        self._lock = threading.RLock()
        self._session_id = ""
        self._next_request_sequence = 0
        self._last_external_request_sequences: dict[str, int] = {}
        self._external_session_id = ""
        self._external_request_id = ""
        self._external_event_sink: Any = None
        self._cycle_event_sequence = 0
        self._leg_accepted = False
        self._action_datagrams = 0
        self._child_event_sink = self._receive_child_event

    @property
    def status(self) -> CycleStatus:
        with self._lock:
            return self._cycle.status

    def start(
        self,
        *,
        cycle_id: str,
        dig_trajectory: Mapping[str, Any],
    ) -> None:
        with self._lock:
            directive = self._cycle.start(
                cycle_id=cycle_id,
                dig_trajectory=dig_trajectory,
            )
            self._session_id = "cycle:%s" % cycle_id
            self._next_request_sequence = 0
            self._dispatch(directive)

    def handle(
        self,
        request: Mapping[str, Any],
        event_sink: Any,
    ) -> None:
        request_type = request.get("type") if isinstance(request, Mapping) else None
        if request_type not in {
            "start_cycle",
            "provide_dump_trajectory",
            "cancel_cycle",
        }:
            self._behavior_executor.handle(request, event_sink)
            return
        expected_fields = {
            "schema_version",
            "type",
            "session_id",
            "seq",
            "request_id",
            "cycle_id",
        }
        if request_type == "start_cycle":
            expected_fields.add("dig_trajectory")
        elif request_type == "provide_dump_trajectory":
            expected_fields.add("dump_trajectory")
        identity = request_identity(request, expected_type=request_type)
        if set(request) != expected_fields:
            self._emit_rejected(identity, event_sink, "BAD_REQUEST", "cycle request fields are invalid")
            return
        if not self._accept_external_request_sequence(identity):
            self._emit_rejected(identity, event_sink, "OUT_OF_ORDER", "request seq is not increasing")
            return
        cycle_id = request.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            self._emit_rejected(identity, event_sink, "BAD_REQUEST", "cycle_id must be non-empty text")
            return
        with self._lock:
            self._external_session_id = identity[0]
            self._external_request_id = identity[2]
            self._external_event_sink = event_sink
            if request_type != "cancel_cycle":
                self._leg_accepted = False
            try:
                if request_type == "start_cycle":
                    self._action_datagrams = 0
                    self.start(
                        cycle_id=cycle_id,
                        dig_trajectory=request["dig_trajectory"],
                    )
                elif request_type == "provide_dump_trajectory":
                    if cycle_id != self._cycle.status.cycle_id:
                        raise ValueError("cycle_id does not match active cycle")
                    self.provide_dump_trajectory(request["dump_trajectory"])
                else:
                    if cycle_id != self._cycle.status.cycle_id:
                        raise ValueError("cycle_id does not match active cycle")
                    self._cancel_active_behavior()
            except (RuntimeError, ValueError) as exc:
                self._emit_rejected(
                    identity,
                    event_sink,
                    "INVALID_CYCLE_STATE",
                    str(exc),
                )
                self._external_event_sink = None

    def provide_dump_trajectory(
        self,
        dump_trajectory: Mapping[str, Any],
    ) -> None:
        with self._lock:
            directive = self._cycle.provide_dump_trajectory(dump_trajectory)
            self._dispatch(directive)

    def observe(self, machine_state: Mapping[str, Any]) -> None:
        self._behavior_executor.observe(machine_state)

    def watchdog(self) -> None:
        self._behavior_executor.watchdog()

    def close(self, *, emit_result: bool = True) -> None:
        self._behavior_executor.close(emit_result=emit_result)

    def _dispatch(self, directive: CycleDirective) -> None:
        sequence = self._next_request_sequence
        self._next_request_sequence = sequence + 1
        request_id = "%s:%s" % (
            self._cycle.status.cycle_id,
            directive.stage.lower(),
        )
        common = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self._session_id,
            "seq": sequence,
            "request_id": request_id,
        }
        if directive.behavior == "Follow":
            request = {
                **common,
                "type": "start_follow",
                "trajectory": copy.deepcopy(directive.trajectory),
            }
        else:
            request = {
                **common,
                "type": "start_fixed_action",
                "behavior": directive.behavior,
            }
        self._behavior_executor.handle(request, self._child_event_sink)

    def _cancel_active_behavior(self) -> None:
        behavior = self._cycle.status.active_behavior
        if not behavior:
            self._cycle.cancel()
            self._emit_cycle_event(
                "result",
                cycle_id=self._cycle.status.cycle_id,
                outcome="CANCELLED",
                reason_code="CANCELLED",
                message="excavation cycle cancelled while quiescent",
                completed_stage=self._cycle.status.stage,
                quiescence_confirmed=True,
                action_datagrams=self._action_datagrams,
            )
            self._external_event_sink = None
            return
        sequence = self._next_request_sequence
        self._next_request_sequence = sequence + 1
        request = {
            "schema_version": SCHEMA_VERSION,
            "type": (
                "cancel_follow"
                if behavior == "Follow"
                else "cancel_fixed_action"
            ),
            "session_id": self._session_id,
            "seq": sequence,
            "request_id": "%s:cancel" % self._cycle.status.cycle_id,
        }
        self._behavior_executor.handle(request, self._child_event_sink)

    def _receive_child_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "accepted":
            with self._lock:
                if not self._leg_accepted:
                    self._leg_accepted = True
                    self._emit_cycle_event(
                        "accepted",
                        cycle_id=self._cycle.status.cycle_id,
                        stage=self._cycle.status.stage,
                    )
                else:
                    self._emit_cycle_event(
                        "feedback",
                        cycle_id=self._cycle.status.cycle_id,
                        stage=self._cycle.status.stage,
                        behavior=self._cycle.status.active_behavior,
                        message="local behavior accepted",
                        action_datagrams=self._action_datagrams,
                    )
            return
        if event_type == "feedback":
            with self._lock:
                self._emit_cycle_event(
                    "feedback",
                    cycle_id=self._cycle.status.cycle_id,
                    stage=self._cycle.status.stage,
                    behavior=self._cycle.status.active_behavior,
                    message="local behavior running",
                    action_datagrams=self._action_datagrams,
                    behavior_feedback={
                        key: copy.deepcopy(value)
                        for key, value in event.items()
                        if key
                        not in {
                            "schema_version",
                            "type",
                            "session_id",
                            "seq",
                            "request_id",
                        }
                    },
                )
            return
        with self._lock:
            completed_stage = self._cycle.status.stage
            if event_type == "rejected":
                reason_code = str(
                    event.get("reason_code") or "CHILD_REJECTED"
                )
                directive = self._cycle.record_behavior_result(
                    outcome="FAILED",
                    reason_code=reason_code,
                    quiescence_confirmed=True,
                )
                if not self._leg_accepted:
                    self._emit_cycle_event(
                        "rejected",
                        cycle_id=self._cycle.status.cycle_id,
                        reason_code=reason_code,
                        message=str(event.get("message") or "child behavior rejected"),
                    )
                    self._external_event_sink = None
                    return
            else:
                directive = self._cycle.record_behavior_result(
                    outcome=str(event.get("outcome") or "FAILED"),
                    reason_code=str(event.get("reason_code") or "CHILD_FAILED"),
                    quiescence_confirmed=event.get("quiescence_confirmed") is True,
                )
                self._action_datagrams += self._event_action_datagrams(event)
            if directive is not None:
                self._dispatch(directive)
                return
            status = self._cycle.status
            if status.stage == "WAITING_FOR_DUMP_TRAJECTORY":
                self._emit_cycle_event(
                    "result",
                    cycle_id=status.cycle_id,
                    outcome="SUCCEEDED",
                    reason_code="DIG_LEG_COMPLETED",
                    message="FollowDig and ExecuteDig completed locally",
                    completed_stage=completed_stage,
                    quiescence_confirmed=True,
                    action_datagrams=self._action_datagrams,
                )
                self._external_event_sink = None
            elif status.terminal:
                self._emit_cycle_event(
                    "result",
                    cycle_id=status.cycle_id,
                    outcome=status.outcome,
                    reason_code=status.reason_code,
                    message=(
                        "excavation cycle completed"
                        if status.outcome == "SUCCEEDED"
                        else "excavation cycle stopped"
                    ),
                    completed_stage=completed_stage,
                    quiescence_confirmed=event.get("quiescence_confirmed") is True,
                    action_datagrams=self._action_datagrams,
                )
                self._external_event_sink = None

    def status_event(self) -> dict[str, Any]:
        return self._behavior_executor.status_event()

    def disconnect(self, event_sink: Any) -> None:
        with self._lock:
            if self._external_event_sink is event_sink:
                if self._behavior_executor.busy:
                    self._behavior_executor.close(emit_result=True)
                self._external_event_sink = None
                return
        self._behavior_executor.disconnect(event_sink)

    def _accept_external_request_sequence(
        self,
        identity: tuple[str, int, str],
    ) -> bool:
        last = self._last_external_request_sequences.get(identity[0])
        if last is not None and identity[1] <= last:
            return False
        self._last_external_request_sequences = {
            **self._last_external_request_sequences,
            identity[0]: identity[1],
        }
        return True

    def _emit_cycle_event(self, event_type: str, **fields: Any) -> None:
        sink = self._external_event_sink
        if sink is None:
            return
        event = {
            "schema_version": SCHEMA_VERSION,
            "type": event_type,
            "session_id": self._external_session_id,
            "seq": self._cycle_event_sequence,
            "request_id": self._external_request_id,
            **fields,
        }
        self._cycle_event_sequence += 1
        sink(event)

    def _emit_rejected(
        self,
        identity: tuple[str, int, str],
        event_sink: Any,
        reason_code: str,
        message: str,
    ) -> None:
        event_sink(
            {
                "schema_version": SCHEMA_VERSION,
                "type": "rejected",
                "session_id": identity[0],
                "seq": self._cycle_event_sequence,
                "request_id": identity[2],
                "reason_code": reason_code,
                "message": message,
            }
        )
        self._cycle_event_sequence += 1

    @staticmethod
    def _event_action_datagrams(event: Mapping[str, Any]) -> int:
        value = event.get("action_datagrams")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value
