"""Atomic STM32 write boundary for resident RL/ACT policy workers.

The sink is the only object allowed to combine policy selection with a serial
write.  Handoff generation, latest safety telemetry, command encoding and the
actual write all share one lock, closing the gap where an old policy could
otherwise write after a newer handoff had already begun.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
import threading
from typing import Callable, Protocol
import uuid

from .resident_commands import Stm32ResidentCommandEncoder
from .resident_motion import (
    ControlMode,
    MotionCandidate,
    PolicyBinding,
    ResidentMotionAuthority,
    ResidentMotionSnapshot,
    ZERO_ACTION,
)


_PASSIVE_REJECTION_REASONS = frozenset(
    {"no_active_policy", "wrong_source", "stale_generation", "wrong_mode"}
)
HANDOFF_SAMPLE_SCHEMA_VERSION = "resident_handoff_sample.v1"
HANDOFF_SAMPLE_LOG_PREFIX = "RESIDENT_HANDOFF_SAMPLE "
_HANDOFF_LOGGER = logging.getLogger("edge_runtime.resident_handoff")
_IDLE_ZERO_KEEPALIVE_NS = 100_000_000


class SerialWriter(Protocol):
    def write(self, payload: bytes) -> int: ...

    def flush(self) -> None: ...


@dataclass(frozen=True)
class ResidentTelemetry:
    """The minimum STM32 evidence needed at the final motion boundary."""

    receive_monotonic_ns: int
    command_rx_seq: int
    command_valid: bool
    command_timed_out: bool
    control_mode: ControlMode | None
    command_action: tuple[float, float, float, float]
    control_enabled: bool
    estop: bool
    sensor_valid: bool
    stm32_alive: bool
    fault_flags: int

    def __post_init__(self) -> None:
        _nonnegative_integer("receive_monotonic_ns", self.receive_monotonic_ns)
        _uint32("command_rx_seq", self.command_rx_seq)
        for name in (
            "command_valid",
            "command_timed_out",
            "control_enabled",
            "estop",
            "sensor_valid",
            "stm32_alive",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.control_mode is not None:
            object.__setattr__(self, "control_mode", ControlMode(self.control_mode))
        object.__setattr__(self, "command_action", _action(self.command_action))
        if (
            isinstance(self.fault_flags, bool)
            or not isinstance(self.fault_flags, int)
            or self.fault_flags < 0
        ):
            raise ValueError("fault_flags must be a nonnegative integer")

    @property
    def command_received(self) -> bool:
        return self.command_valid or self.command_timed_out


@dataclass(frozen=True)
class ResidentWriteResult:
    accepted: bool
    write_performed: bool
    reason: str
    command_seq: int | None
    mode: ControlMode | None
    effective_action: tuple[float, float, float, float]


@dataclass(frozen=True)
class ResidentHandoffSample:
    """Append-only evidence for one physical policy-to-policy handoff."""

    runtime_id: str
    generation: int
    source_binding: PolicyBinding
    target_binding: PolicyBinding
    terminal_zero_command_seq: int
    terminal_zero_ack_monotonic_ns: int
    target_zero_command_seq: int
    target_zero_ack_monotonic_ns: int
    first_nonzero_command_seq: int
    first_nonzero_action: tuple[float, float, float, float]
    first_nonzero_write_monotonic_ns: int
    first_nonzero_ack_monotonic_ns: int

    def as_payload(self) -> dict[str, object]:
        terminal_ack = self.terminal_zero_ack_monotonic_ns
        target_ack = self.target_zero_ack_monotonic_ns
        first_write = self.first_nonzero_write_monotonic_ns
        first_ack = self.first_nonzero_ack_monotonic_ns
        return {
            "schema_version": HANDOFF_SAMPLE_SCHEMA_VERSION,
            "runtime_id": self.runtime_id,
            "generation": self.generation,
            "from_source": self.source_binding.source,
            "from_mode": self.source_binding.mode.value,
            "to_source": self.target_binding.source,
            "to_mode": self.target_binding.mode.value,
            "terminal_zero_command_seq": self.terminal_zero_command_seq,
            "terminal_zero_ack_monotonic_ns": terminal_ack,
            "target_zero_command_seq": self.target_zero_command_seq,
            "target_zero_ack_monotonic_ns": target_ack,
            "first_nonzero_command_seq": self.first_nonzero_command_seq,
            "first_nonzero_action": list(self.first_nonzero_action),
            "first_nonzero_write_monotonic_ns": first_write,
            "first_nonzero_ack_monotonic_ns": first_ack,
            "zero_claim_ms": _elapsed_ms(terminal_ack, target_ack),
            "policy_ready_wait_ms": _elapsed_ms(target_ack, first_write),
            "first_command_ack_ms": _elapsed_ms(first_write, first_ack),
            "latency_ms": _elapsed_ms(terminal_ack, first_ack),
        }


@dataclass(frozen=True)
class _PendingAck:
    generation: int
    mode: ControlMode
    purpose: str
    command_seq: int


@dataclass(frozen=True)
class _CandidateLease:
    binding: PolicyBinding
    generation: int
    valid_until_monotonic_ns: int


@dataclass(frozen=True)
class _HandoffMeasurement:
    runtime_id: str
    generation: int
    source_binding: PolicyBinding
    target_binding: PolicyBinding
    terminal_zero_command_seq: int
    terminal_zero_acknowledged_monotonic_ns: int
    target_zero_command_seq: int | None = None
    target_zero_acknowledged_monotonic_ns: int | None = None


@dataclass(frozen=True)
class _PendingFirstNonzeroAck:
    generation: int
    mode: ControlMode
    command_seq: int
    action: tuple[float, float, float, float]
    write_monotonic_ns: int


class ResidentCommandSink:
    """Own a single serial writer and arbitrate all resident policy commands."""

    def __init__(
        self,
        serial_writer: SerialWriter,
        *,
        max_state_age_ms: float,
        max_future_skew_ms: float = 5.0,
        runtime_id: str | None = None,
        handoff_sample_sink: Callable[[ResidentHandoffSample], None] | None = None,
    ) -> None:
        if not math.isfinite(max_state_age_ms) or max_state_age_ms <= 0.0:
            raise ValueError("state age limit must be finite and positive")
        self._serial = serial_writer
        self._max_state_age_ns = int(max_state_age_ms * 1_000_000)
        self._authority = ResidentMotionAuthority(
            max_future_skew_ms=max_future_skew_ms
        )
        self._encoder = Stm32ResidentCommandEncoder()
        self._runtime_id = _runtime_id(
            uuid.uuid4().hex if runtime_id is None else runtime_id
        )
        self._handoff_sample_sink = handoff_sample_sink or _log_handoff_sample
        self._latest_telemetry: ResidentTelemetry | None = None
        self._pending_ack: _PendingAck | None = None
        self._candidate_lease: _CandidateLease | None = None
        self._handoff_measurement: _HandoffMeasurement | None = None
        self._pending_first_nonzero_ack: _PendingFirstNonzeroAck | None = None
        self._last_command_write_ns: int | None = None
        self._synchronized = False
        self._unsafe_zero_latched = False
        self._terminally_disarmed = False
        self._terminal_error: str | None = None
        self._lock = threading.Lock()

    def initialize(self, frame: ResidentTelemetry) -> int:
        with self._lock:
            if self._synchronized:
                raise RuntimeError("resident command sink is already initialized")
            next_sequence = self._encoder.synchronize(
                command_rx_seq=frame.command_rx_seq,
                command_received=frame.command_received,
            )
            self._latest_telemetry = frame
            self._synchronized = True
            return next_sequence

    def snapshot(self) -> ResidentMotionSnapshot:
        return self._authority.snapshot()

    @property
    def is_operational(self) -> bool:
        with self._lock:
            return (
                self._synchronized
                and not self._terminally_disarmed
                and self._terminal_error is None
            )

    def request_handoff(
        self,
        binding: PolicyBinding,
        *,
        now_monotonic_ns: int,
    ) -> int:
        now = _nonnegative_integer("now_monotonic_ns", now_monotonic_ns)
        with self._lock:
            self._require_operational()
            self._cancel_handoff_measurement_locked()
            generation = self._authority.request_handoff(
                binding,
                now_monotonic_ns=now,
            )
            self._candidate_lease = None
            self._unsafe_zero_latched = False
            self._issue_pending_zero_locked(now)
            return generation

    def request_stop(self, *, now_monotonic_ns: int) -> int:
        now = _nonnegative_integer("now_monotonic_ns", now_monotonic_ns)
        with self._lock:
            self._require_operational()
            generation = self._authority.request_stop(now_monotonic_ns=now)
            self._candidate_lease = None
            self._cancel_handoff_measurement_locked()
            self._unsafe_zero_latched = True
            self._pending_ack = None
            self._issue_pending_zero_locked(now)
            return generation

    def request_source_stop(
        self,
        binding: PolicyBinding,
        *,
        generation: int,
        now_monotonic_ns: int,
    ) -> bool:
        """Stop only if ``binding`` still belongs to the current generation.

        A delayed or malformed output from a policy that has already lost
        authority must not cancel the newer policy handoff.  The stale worker
        is rejected locally while the current authority state remains intact.
        """

        current_binding = PolicyBinding(binding.source, binding.mode)
        current_generation = _nonnegative_integer("generation", generation)
        now = _nonnegative_integer("now_monotonic_ns", now_monotonic_ns)
        with self._lock:
            self._require_operational()
            snapshot = self._authority.snapshot()
            if snapshot.generation != current_generation or current_binding not in (
                snapshot.active_binding,
                snapshot.target_binding,
            ):
                return False
            self._authority.request_stop(now_monotonic_ns=now)
            self._candidate_lease = None
            self._cancel_handoff_measurement_locked()
            self._unsafe_zero_latched = True
            self._pending_ack = None
            self._issue_pending_zero_locked(now)
            return True

    def observe_telemetry(self, frame: ResidentTelemetry) -> None:
        with self._lock:
            self._require_initialized()
            self._observe_telemetry_locked(frame)

    def invalidate_telemetry(
        self,
        *,
        receive_monotonic_ns: int,
        stm32_alive: bool,
    ) -> None:
        """Immediately revoke motion when one expected telemetry row is invalid."""

        now = _nonnegative_integer(
            "receive_monotonic_ns",
            receive_monotonic_ns,
        )
        if not isinstance(stm32_alive, bool):
            raise ValueError("stm32_alive must be boolean")
        with self._lock:
            self._require_initialized()
            previous = self._latest_telemetry
            if previous is None:
                raise RuntimeError("resident telemetry is unavailable")
            self._observe_telemetry_locked(
                replace(
                    previous,
                    receive_monotonic_ns=now,
                    sensor_valid=False,
                    stm32_alive=stm32_alive,
                )
            )

    def _observe_telemetry_locked(self, frame: ResidentTelemetry) -> None:
        previous = self._latest_telemetry
        if (
            previous is not None
            and frame.receive_monotonic_ns < previous.receive_monotonic_ns
        ):
            return
        self._latest_telemetry = frame
        if self._terminally_disarmed or self._terminal_error is not None:
            return

        if self._pending_ack is not None and self._ack_matches_locked(frame):
            pending = self._pending_ack
            authority_before_ack = self._authority.snapshot()
            handoff_target_exists = (
                pending.purpose == "terminal_source_zero"
                and authority_before_ack.active_binding is not None
                and authority_before_ack.target_binding is not None
            )
            self._authority.acknowledge_zero(
                generation=pending.generation,
                mode=pending.mode,
                action=ZERO_ACTION,
                acknowledged_monotonic_ns=frame.receive_monotonic_ns,
            )
            if handoff_target_exists:
                if (
                    authority_before_ack.active_binding is None
                    or authority_before_ack.target_binding is None
                ):
                    raise RuntimeError("handoff bindings disappeared before zero ACK")
                self._handoff_measurement = _HandoffMeasurement(
                    runtime_id=self._runtime_id,
                    generation=pending.generation,
                    source_binding=authority_before_ack.active_binding,
                    target_binding=authority_before_ack.target_binding,
                    terminal_zero_command_seq=pending.command_seq,
                    terminal_zero_acknowledged_monotonic_ns=(
                        frame.receive_monotonic_ns
                    ),
                )
            elif pending.purpose == "target_mode_claim":
                measurement = self._handoff_measurement
                if (
                    measurement is not None
                    and measurement.generation == pending.generation
                ):
                    self._handoff_measurement = replace(
                        measurement,
                        target_zero_command_seq=pending.command_seq,
                        target_zero_acknowledged_monotonic_ns=(
                            frame.receive_monotonic_ns
                        ),
                    )
            self._pending_ack = None
            if self._authority.pending_zero() is not None:
                self._issue_pending_zero_locked(frame.receive_monotonic_ns)

        if (
            self._pending_first_nonzero_ack is not None
            and self._first_nonzero_ack_matches_locked(frame)
        ):
            pending_motion = self._pending_first_nonzero_ack
            measurement = self._handoff_measurement
            if (
                measurement is None
                or measurement.generation != pending_motion.generation
            ):
                raise RuntimeError("handoff measurement state is inconsistent")
            self._authority.record_handoff_latency(
                generation=measurement.generation,
                terminal_zero_acknowledged_monotonic_ns=(
                    measurement.terminal_zero_acknowledged_monotonic_ns
                ),
                first_nonzero_acknowledged_monotonic_ns=(
                    frame.receive_monotonic_ns
                ),
            )
            self._emit_handoff_sample_locked(
                measurement,
                pending_motion,
                first_nonzero_ack_monotonic_ns=frame.receive_monotonic_ns,
            )
            self._cancel_handoff_measurement_locked()

        snapshot = self._authority.snapshot()
        if snapshot.active_binding is None or self._pending_ack is not None:
            return
        if self._safety_permits_locked(
            now_monotonic_ns=frame.receive_monotonic_ns,
            expected_mode=snapshot.active_binding.mode,
        ):
            self._unsafe_zero_latched = False
            return
        if not self._unsafe_zero_latched:
            self._revoke_for_safety_locked(
                frame.receive_monotonic_ns,
                reason="unsafe_telemetry",
            )

    def submit_candidate(
        self,
        candidate: MotionCandidate,
        *,
        now_monotonic_ns: int,
    ) -> ResidentWriteResult:
        now = _nonnegative_integer("now_monotonic_ns", now_monotonic_ns)
        with self._lock:
            self._require_initialized()
            if self._terminal_error is not None:
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason="sink_faulted",
                    command_seq=None,
                    mode=None,
                    effective_action=ZERO_ACTION,
                )
            if self._terminally_disarmed:
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason="terminally_disarmed",
                    command_seq=None,
                    mode=None,
                    effective_action=ZERO_ACTION,
                )
            snapshot = self._authority.snapshot()
            expected_mode = (
                None
                if snapshot.active_binding is None
                else snapshot.active_binding.mode
            )
            safety_permits = (
                expected_mode is not None
                and self._safety_permits_locked(
                    now_monotonic_ns=now,
                    expected_mode=expected_mode,
                )
            )
            decision = self._authority.route(
                candidate,
                now_monotonic_ns=now,
                safety_permits_motion=safety_permits,
            )

            if (
                self._pending_ack is not None
                or decision.mode is None
                or decision.reason in _PASSIVE_REJECTION_REASONS
            ):
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason=decision.reason,
                    command_seq=None,
                    mode=decision.mode,
                    effective_action=ZERO_ACTION,
                )

            if (
                decision.accepted
                and self._pending_first_nonzero_ack is not None
            ):
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason="handoff_first_action_ack_pending",
                    command_seq=None,
                    mode=decision.mode,
                    effective_action=ZERO_ACTION,
                )

            if decision.reason == "safety_rejected":
                return self._revoke_for_safety_locked(
                    now,
                    reason="safety_rejected",
                )

            result = self._write_locked(
                mode=decision.mode,
                action=decision.effective_action,
                monotonic_ns=now,
                reason=decision.reason,
                accepted=decision.accepted,
            )
            if decision.accepted and decision.effective_action != ZERO_ACTION:
                self._candidate_lease = _CandidateLease(
                    binding=PolicyBinding(candidate.source, candidate.mode),
                    generation=candidate.generation,
                    valid_until_monotonic_ns=candidate.valid_until_monotonic_ns,
                )
                measurement = self._handoff_measurement
                if (
                    measurement is not None
                    and measurement.generation == candidate.generation
                    and self._pending_first_nonzero_ack is None
                ):
                    if result.command_seq is None:
                        raise RuntimeError(
                            "accepted handoff action has no command sequence"
                        )
                    self._pending_first_nonzero_ack = _PendingFirstNonzeroAck(
                        generation=candidate.generation,
                        mode=candidate.mode,
                        command_seq=result.command_seq,
                        action=result.effective_action,
                        write_monotonic_ns=now,
                    )
            elif result.effective_action == ZERO_ACTION:
                self._candidate_lease = None
                if not decision.accepted:
                    self._cancel_handoff_measurement_locked()
            self._unsafe_zero_latched = not decision.accepted
            return result

    def tick(self, *, now_monotonic_ns: int) -> ResidentWriteResult | None:
        """Enforce the state-silence watchdog even when no candidate arrives."""

        now = _nonnegative_integer("now_monotonic_ns", now_monotonic_ns)
        with self._lock:
            self._require_initialized()
            if self._terminally_disarmed or self._terminal_error is not None:
                return None
            snapshot = self._authority.snapshot()
            lease = self._candidate_lease
            if lease is not None:
                if (
                    snapshot.active_binding != lease.binding
                    or snapshot.generation != lease.generation
                ):
                    self._candidate_lease = None
                elif now > lease.valid_until_monotonic_ns:
                    self._authority.request_stop(now_monotonic_ns=now)
                    self._candidate_lease = None
                    self._cancel_handoff_measurement_locked()
                    self._unsafe_zero_latched = True
                    self._pending_ack = None
                    return self._issue_pending_zero_locked(
                        now,
                        write_reason="candidate_lease_expired",
                    )
            if (
                snapshot.active_binding is None
                or self._pending_ack is not None
                or self._unsafe_zero_latched
            ):
                return None
            if self._safety_permits_locked(
                now_monotonic_ns=now,
                expected_mode=snapshot.active_binding.mode,
            ):
                last_write = self._last_command_write_ns
                if (
                    lease is None
                    and self._pending_first_nonzero_ack is None
                    and last_write is not None
                    and now - last_write >= _IDLE_ZERO_KEEPALIVE_NS
                ):
                    return self._write_locked(
                        mode=snapshot.active_binding.mode,
                        action=ZERO_ACTION,
                        monotonic_ns=now,
                        reason="active_policy_idle_keepalive",
                        accepted=False,
                    )
                return None
            return self._revoke_for_safety_locked(
                now,
                reason="state_watchdog",
            )

    def terminal_disarm(
        self,
        *,
        now_monotonic_ns: int,
    ) -> ResidentWriteResult:
        """Permanently prevent later motion and make the final write zero."""

        now = _nonnegative_integer("now_monotonic_ns", now_monotonic_ns)
        with self._lock:
            self._require_initialized()
            if self._terminally_disarmed or self._terminal_error is not None:
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason=(
                        "sink_faulted"
                        if self._terminal_error is not None
                        else "terminally_disarmed"
                    ),
                    command_seq=None,
                    mode=None,
                    effective_action=ZERO_ACTION,
                )
            snapshot = self._authority.snapshot()
            mode = (
                snapshot.active_binding.mode
                if snapshot.active_binding is not None
                else (
                    None
                    if self._pending_ack is None
                    else self._pending_ack.mode
                )
            )
            self._authority.request_stop(now_monotonic_ns=now)
            self._candidate_lease = None
            self._cancel_handoff_measurement_locked()
            self._pending_ack = None
            self._terminally_disarmed = True
            self._unsafe_zero_latched = True
            if mode is None:
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason="terminal_disarm",
                    command_seq=None,
                    mode=None,
                    effective_action=ZERO_ACTION,
                )
            return self._write_locked(
                mode=mode,
                action=ZERO_ACTION,
                monotonic_ns=now,
                reason="terminal_disarm",
                accepted=False,
            )

    def _require_initialized(self) -> None:
        if not self._synchronized:
            raise RuntimeError("resident command sequence is not initialized")

    def _require_operational(self) -> None:
        self._require_initialized()
        if self._terminal_error is not None:
            raise RuntimeError(f"resident command sink is faulted: {self._terminal_error}")
        if self._terminally_disarmed:
            raise RuntimeError("resident command sink is terminally disarmed")

    def _issue_pending_zero_locked(
        self,
        monotonic_ns: int,
        *,
        write_reason: str | None = None,
    ) -> ResidentWriteResult | None:
        pending = self._authority.pending_zero()
        if pending is None:
            self._pending_ack = None
            return None
        result = self._write_locked(
            mode=pending.mode,
            action=ZERO_ACTION,
            monotonic_ns=monotonic_ns,
            reason=pending.purpose if write_reason is None else write_reason,
            accepted=False,
        )
        if result.command_seq is None:
            raise RuntimeError("pending zero did not produce a command sequence")
        self._pending_ack = _PendingAck(
            generation=pending.generation,
            mode=pending.mode,
            purpose=pending.purpose,
            command_seq=result.command_seq,
        )
        return result

    def _revoke_for_safety_locked(
        self,
        monotonic_ns: int,
        *,
        reason: str,
    ) -> ResidentWriteResult:
        """Revoke the generation before issuing a safety-triggered zero."""

        self._authority.request_stop(now_monotonic_ns=monotonic_ns)
        self._candidate_lease = None
        self._cancel_handoff_measurement_locked()
        self._unsafe_zero_latched = True
        self._pending_ack = None
        result = self._issue_pending_zero_locked(
            monotonic_ns,
            write_reason=reason,
        )
        if result is None:
            raise RuntimeError("safety revocation did not produce a zero command")
        return result

    def _ack_matches_locked(self, frame: ResidentTelemetry) -> bool:
        pending = self._pending_ack
        if pending is None:
            return False
        return (
            frame.command_rx_seq == pending.command_seq
            and frame.command_valid
            and not frame.command_timed_out
            and frame.control_mode is pending.mode
            and frame.command_action == ZERO_ACTION
        )

    def _first_nonzero_ack_matches_locked(
        self,
        frame: ResidentTelemetry,
    ) -> bool:
        pending = self._pending_first_nonzero_ack
        if pending is None:
            return False
        return (
            frame.command_rx_seq == pending.command_seq
            and frame.command_valid
            and not frame.command_timed_out
            and frame.control_mode is pending.mode
        )

    def _emit_handoff_sample_locked(
        self,
        measurement: _HandoffMeasurement,
        pending: _PendingFirstNonzeroAck,
        *,
        first_nonzero_ack_monotonic_ns: int,
    ) -> None:
        if (
            measurement.target_zero_command_seq is None
            or measurement.target_zero_acknowledged_monotonic_ns is None
        ):
            raise RuntimeError("handoff target zero ACK evidence is incomplete")
        sample = ResidentHandoffSample(
            runtime_id=measurement.runtime_id,
            generation=measurement.generation,
            source_binding=measurement.source_binding,
            target_binding=measurement.target_binding,
            terminal_zero_command_seq=measurement.terminal_zero_command_seq,
            terminal_zero_ack_monotonic_ns=(
                measurement.terminal_zero_acknowledged_monotonic_ns
            ),
            target_zero_command_seq=measurement.target_zero_command_seq,
            target_zero_ack_monotonic_ns=(
                measurement.target_zero_acknowledged_monotonic_ns
            ),
            first_nonzero_command_seq=pending.command_seq,
            first_nonzero_action=pending.action,
            first_nonzero_write_monotonic_ns=pending.write_monotonic_ns,
            first_nonzero_ack_monotonic_ns=first_nonzero_ack_monotonic_ns,
        )
        try:
            self._handoff_sample_sink(sample)
        except Exception:
            # Evidence persistence must not interrupt the final serial boundary.
            _HANDOFF_LOGGER.exception("resident handoff sample sink failed")

    def _cancel_handoff_measurement_locked(self) -> None:
        self._handoff_measurement = None
        self._pending_first_nonzero_ack = None

    def _safety_permits_locked(
        self,
        *,
        now_monotonic_ns: int,
        expected_mode: ControlMode,
    ) -> bool:
        frame = self._latest_telemetry
        if frame is None:
            return False
        if now_monotonic_ns < frame.receive_monotonic_ns:
            return False
        if now_monotonic_ns - frame.receive_monotonic_ns > self._max_state_age_ns:
            return False
        return (
            frame.control_enabled
            and not frame.estop
            and frame.sensor_valid
            and frame.stm32_alive
            and frame.fault_flags == 0
            and frame.command_valid
            and not frame.command_timed_out
            and frame.control_mode is expected_mode
        )

    def _write_locked(
        self,
        *,
        mode: ControlMode,
        action: tuple[float, float, float, float],
        monotonic_ns: int,
        reason: str,
        accepted: bool,
    ) -> ResidentWriteResult:
        sequence = self._encoder.next_sequence
        try:
            payload = self._encoder.encode(
                mode=mode,
                action=action,
                monotonic_ns=monotonic_ns,
            )
            written = self._serial.write(payload)
            if written != len(payload):
                raise OSError(f"short serial write: {written}/{len(payload)} bytes")
            self._serial.flush()
            self._last_command_write_ns = monotonic_ns
        except Exception as exc:
            self._cancel_handoff_measurement_locked()
            self._terminal_error = f"{type(exc).__name__}: {exc}"
            self._terminally_disarmed = True
            raise
        return ResidentWriteResult(
            accepted=accepted,
            write_performed=True,
            reason=reason,
            command_seq=sequence,
            mode=mode,
            effective_action=action,
        )


def _action(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not isinstance(values, tuple) or len(values) != 4:
        raise ValueError("command_action must be a four-axis tuple")
    action_values = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("command_action must contain finite numeric values")
        action_values.append(float(value))
    action = tuple(action_values)
    if not all(math.isfinite(value) for value in action):
        raise ValueError("command_action must contain finite values")
    return action  # type: ignore[return-value]


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _uint32(name: str, value: int) -> int:
    number = _nonnegative_integer(name, value)
    if number > 0xFFFFFFFF:
        raise ValueError(f"{name} must be a uint32")
    return number


def _runtime_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise ValueError("runtime_id must be a non-empty string of at most 128 chars")
    return value.strip()


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    if end_ns < start_ns:
        raise RuntimeError("handoff evidence timestamps are not monotonic")
    return (end_ns - start_ns) / 1_000_000.0


def _log_handoff_sample(sample: ResidentHandoffSample) -> None:
    payload = json.dumps(
        sample.as_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    _HANDOFF_LOGGER.info("%s%s", HANDOFF_SAMPLE_LOG_PREFIX, payload)
