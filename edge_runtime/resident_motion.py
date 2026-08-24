"""Policy-independent motion arbitration for a resident Orin runtime.

This module deliberately does not own serial, CUDA, ONNX Runtime, cameras, or
network transport.  It concentrates the one rule all concrete adapters must
share: one policy source and one action semantic may be active at a time, and
every semantic switch passes through acknowledged zero commands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading


ACTION_ORDER = ("boom", "stick", "bucket", "swing")
ZERO_ACTION = (0.0, 0.0, 0.0, 0.0)
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class ControlMode(str, Enum):
    """The two nonzero command semantics supported by unified STM32 firmware."""

    MANUAL_ACTION = "manual_action"
    VELOCITY_REFERENCE = "velocity_reference"


class HandoffPhase(str, Enum):
    IDLE = "idle"
    TERMINAL_ZERO_PENDING = "terminal_zero_pending"
    TARGET_ZERO_PENDING = "target_zero_pending"
    ACTIVE = "active"


@dataclass(frozen=True)
class PolicyBinding:
    source: str
    mode: ControlMode

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("policy source must be a non-empty string")
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "mode", ControlMode(self.mode))


@dataclass(frozen=True)
class MotionCandidate:
    """One typed candidate produced by a resident policy worker."""

    source: str
    generation: int
    mode: ControlMode
    action: tuple[float, float, float, float]
    created_monotonic_ns: int
    valid_until_monotonic_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("candidate source must be a non-empty string")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
            or self.generation > UINT64_MAX
        ):
            raise ValueError("candidate generation must be a uint64")
        if len(self.action) != len(ACTION_ORDER):
            raise ValueError("candidate action must contain four named axes")
        for name, value in (
            ("created_monotonic_ns", self.created_monotonic_ns),
            ("valid_until_monotonic_ns", self.valid_until_monotonic_ns),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > UINT64_MAX
            ):
                raise ValueError(f"{name} must be a uint64")
        if self.valid_until_monotonic_ns < self.created_monotonic_ns:
            raise ValueError("candidate validity must not end before creation")
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "mode", ControlMode(self.mode))
        object.__setattr__(self, "action", tuple(float(value) for value in self.action))


@dataclass(frozen=True)
class PendingZero:
    generation: int
    mode: ControlMode
    purpose: str


@dataclass(frozen=True)
class RoutedCommand:
    generation: int
    mode: ControlMode | None
    requested_action: tuple[float, float, float, float]
    effective_action: tuple[float, float, float, float]
    accepted: bool
    reason: str


@dataclass(frozen=True)
class ResidentMotionSnapshot:
    generation: int
    phase: HandoffPhase
    active_binding: PolicyBinding | None
    target_binding: PolicyBinding | None
    handoff_requested_monotonic_ns: int | None
    last_handoff_latency_ms: float | None


@dataclass(frozen=True)
class _AuthorityState:
    generation: int = 0
    phase: HandoffPhase = HandoffPhase.IDLE
    active_binding: PolicyBinding | None = None
    target_binding: PolicyBinding | None = None
    handoff_requested_monotonic_ns: int | None = None
    last_handoff_latency_ms: float | None = None


class ResidentMotionAuthority:
    """Serialize policy ownership and make semantic handoffs fail closed.

    Concrete serial adapters must turn :meth:`pending_zero` into the matching
    STM32 wire schema and call :meth:`acknowledge_zero` only after telemetry
    proves that exact zero command was accepted.  Policy workers never receive
    a serial handle and submit only :class:`MotionCandidate` values.
    """

    def __init__(self, *, max_future_skew_ms: float = 5.0) -> None:
        if not math.isfinite(max_future_skew_ms) or max_future_skew_ms < 0.0:
            raise ValueError("future skew limit must be finite and nonnegative")
        self._max_future_skew_ns = int(max_future_skew_ms * 1_000_000)
        self._state = _AuthorityState()
        self._lock = threading.Lock()

    def snapshot(self) -> ResidentMotionSnapshot:
        with self._lock:
            state = self._state
        return ResidentMotionSnapshot(
            generation=state.generation,
            phase=state.phase,
            active_binding=state.active_binding,
            target_binding=state.target_binding,
            handoff_requested_monotonic_ns=state.handoff_requested_monotonic_ns,
            last_handoff_latency_ms=state.last_handoff_latency_ms,
        )

    def request_handoff(
        self,
        binding: PolicyBinding,
        *,
        now_monotonic_ns: int,
    ) -> int:
        target = PolicyBinding(binding.source, binding.mode)
        now = _monotonic_ns(now_monotonic_ns)
        with self._lock:
            generation = self._state.generation + 1
            phase = (
                HandoffPhase.TERMINAL_ZERO_PENDING
                if self._state.active_binding is not None
                else HandoffPhase.TARGET_ZERO_PENDING
            )
            self._state = replace(
                self._state,
                generation=generation,
                phase=phase,
                target_binding=target,
                handoff_requested_monotonic_ns=now,
            )
            return generation

    def request_stop(self, *, now_monotonic_ns: int) -> int:
        now = _monotonic_ns(now_monotonic_ns)
        with self._lock:
            generation = self._state.generation + 1
            if self._state.active_binding is None:
                self._state = _AuthorityState(
                    generation=generation,
                    last_handoff_latency_ms=self._state.last_handoff_latency_ms,
                )
            else:
                self._state = replace(
                    self._state,
                    generation=generation,
                    phase=HandoffPhase.TERMINAL_ZERO_PENDING,
                    target_binding=None,
                    handoff_requested_monotonic_ns=now,
                )
            return generation

    def pending_zero(self) -> PendingZero | None:
        with self._lock:
            state = self._state
            if state.phase is HandoffPhase.TERMINAL_ZERO_PENDING:
                binding = state.active_binding
                purpose = "terminal_source_zero"
            elif state.phase is HandoffPhase.TARGET_ZERO_PENDING:
                binding = state.target_binding
                purpose = "target_mode_claim"
            else:
                return None
            if binding is None:
                raise RuntimeError("handoff phase has no matching policy binding")
            return PendingZero(
                generation=state.generation,
                mode=binding.mode,
                purpose=purpose,
            )

    def acknowledge_zero(
        self,
        *,
        generation: int,
        mode: ControlMode,
        action: tuple[float, float, float, float],
        acknowledged_monotonic_ns: int,
    ) -> None:
        acknowledged = _monotonic_ns(acknowledged_monotonic_ns)
        observed_mode = ControlMode(mode)
        observed_action = tuple(float(value) for value in action)
        if len(observed_action) != len(ACTION_ORDER) or observed_action != ZERO_ACTION:
            raise ValueError("handoff acknowledgement action must be exactly zero")
        with self._lock:
            state = self._state
            if generation != state.generation:
                raise ValueError("zero acknowledgement is not for current handoff generation")
            if state.handoff_requested_monotonic_ns is None:
                raise ValueError("no handoff is awaiting a zero acknowledgement")
            if acknowledged < state.handoff_requested_monotonic_ns:
                raise ValueError("zero acknowledgement predates the handoff request")
            if state.phase is HandoffPhase.TERMINAL_ZERO_PENDING:
                expected = state.active_binding
            elif state.phase is HandoffPhase.TARGET_ZERO_PENDING:
                expected = state.target_binding
            else:
                raise ValueError("no handoff zero is pending")
            if expected is None or observed_mode is not expected.mode:
                raise ValueError("zero acknowledgement does not match pending handoff mode")

            if state.phase is HandoffPhase.TERMINAL_ZERO_PENDING:
                if state.target_binding is None:
                    self._state = _AuthorityState(
                        generation=state.generation,
                        last_handoff_latency_ms=state.last_handoff_latency_ms,
                    )
                else:
                    self._state = replace(
                        state,
                        phase=HandoffPhase.TARGET_ZERO_PENDING,
                        active_binding=None,
                    )
                return

            target = state.target_binding
            if target is None:
                raise RuntimeError("target zero has no target policy binding")
            self._state = replace(
                state,
                phase=HandoffPhase.ACTIVE,
                active_binding=target,
                target_binding=None,
                handoff_requested_monotonic_ns=None,
                last_handoff_latency_ms=state.last_handoff_latency_ms,
            )

    def record_handoff_latency(
        self,
        *,
        generation: int,
        terminal_zero_acknowledged_monotonic_ns: int,
        first_nonzero_acknowledged_monotonic_ns: int,
    ) -> None:
        """Record the physical policy handoff interval proven by STM32 ACKs.

        Zero-claim completion only makes the target source eligible; it is not
        evidence that the new source has produced motion.  The serial owner
        therefore records latency only after it observes the first nonzero
        command from that source echoed by STM32 telemetry.
        """

        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or not 0 <= generation <= UINT64_MAX
        ):
            raise ValueError("handoff generation must be a uint64")
        started = _monotonic_ns(
            terminal_zero_acknowledged_monotonic_ns
        )
        completed = _monotonic_ns(
            first_nonzero_acknowledged_monotonic_ns
        )
        if completed < started:
            raise ValueError("first nonzero acknowledgement predates terminal zero")
        with self._lock:
            state = self._state
            if generation != state.generation:
                raise ValueError("handoff latency is not for the current generation")
            if state.phase is not HandoffPhase.ACTIVE or state.active_binding is None:
                raise ValueError("handoff target is not active")
            self._state = replace(
                state,
                last_handoff_latency_ms=_elapsed_ms(started, completed),
            )

    def route(
        self,
        candidate: MotionCandidate,
        *,
        now_monotonic_ns: int,
        safety_permits_motion: bool,
    ) -> RoutedCommand:
        now = _monotonic_ns(now_monotonic_ns)
        if not isinstance(safety_permits_motion, bool):
            raise ValueError("safety_permits_motion must be boolean")
        with self._lock:
            state = self._state

        if state.phase is not HandoffPhase.ACTIVE or state.active_binding is None:
            mode = _zero_mode_for_state(state)
            reason = (
                "no_active_policy"
                if state.phase is HandoffPhase.IDLE
                else "handoff_in_progress"
            )
            return _zero_decision(state.generation, mode, candidate.action, reason)

        binding = state.active_binding
        if candidate.source != binding.source:
            return _zero_decision(
                state.generation, binding.mode, candidate.action, "wrong_source"
            )
        if candidate.generation != state.generation:
            return _zero_decision(
                state.generation,
                binding.mode,
                candidate.action,
                "stale_generation",
            )
        if candidate.mode is not binding.mode:
            return _zero_decision(
                state.generation, binding.mode, candidate.action, "wrong_mode"
            )
        if not safety_permits_motion:
            return _zero_decision(
                state.generation,
                binding.mode,
                candidate.action,
                "safety_rejected",
            )
        if now > candidate.valid_until_monotonic_ns:
            return _zero_decision(
                state.generation, binding.mode, candidate.action, "action_expired"
            )
        if candidate.created_monotonic_ns > now + self._max_future_skew_ns:
            return _zero_decision(
                state.generation,
                binding.mode,
                candidate.action,
                "action_from_future",
            )
        if not all(math.isfinite(value) for value in candidate.action):
            return _zero_decision(
                state.generation, binding.mode, candidate.action, "invalid_action"
            )
        if binding.mode is ControlMode.MANUAL_ACTION and not all(
            -1.000001 <= value <= 1.000001 for value in candidate.action
        ):
            return _zero_decision(
                state.generation,
                binding.mode,
                candidate.action,
                "invalid_manual_action",
            )
        return RoutedCommand(
            generation=state.generation,
            mode=binding.mode,
            requested_action=candidate.action,
            effective_action=candidate.action,
            accepted=True,
            reason="accepted",
        )


def _zero_mode_for_state(state: _AuthorityState) -> ControlMode | None:
    if state.phase is HandoffPhase.TERMINAL_ZERO_PENDING:
        return None if state.active_binding is None else state.active_binding.mode
    if state.phase is HandoffPhase.TARGET_ZERO_PENDING:
        return None if state.target_binding is None else state.target_binding.mode
    return None if state.active_binding is None else state.active_binding.mode


def _zero_decision(
    generation: int,
    mode: ControlMode | None,
    requested_action: tuple[float, float, float, float],
    reason: str,
) -> RoutedCommand:
    return RoutedCommand(
        generation=generation,
        mode=mode,
        requested_action=tuple(float(value) for value in requested_action),
        effective_action=ZERO_ACTION,
        accepted=False,
        reason=reason,
    )


def _monotonic_ns(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("monotonic timestamp must be a nonnegative integer")
    return value


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0
