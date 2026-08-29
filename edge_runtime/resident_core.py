"""One resident motion owner shared by RL and imitation-learning workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Callable

from .resident_action_audit import (
    ResidentActionAuditSink,
    emit_action_audit,
)
from .resident_ingress import (
    ResidentPolicyCandidateAdapter,
    ResidentVelocityActionAdapter,
)
from .resident_protocol import decode_motion_candidate
from .resident_motion import (
    ControlMode,
    HandoffPhase,
    PolicyBinding,
    ResidentMotionSnapshot,
    ZERO_ACTION,
)
from .resident_sink import (
    ResidentCommandSink,
    ResidentTelemetry,
    ResidentWriteResult,
    SerialWriter,
)


RL_BINDING = PolicyBinding("rl_follow", ControlMode.VELOCITY_REFERENCE)
ACT_BINDING = PolicyBinding("act_dig", ControlMode.MANUAL_ACTION)
MAX_ACT_SEGMENT_STEPS = 2000
MIN_MISSION_LEASE_MS = 500
MAX_MISSION_LEASE_MS = 5000
ACT_NOMINAL_STEP_PERIOD_MS = 100.0
DEFAULT_ACT_ACTION_CHUNK_STEPS = 10
DEFAULT_ACT_EARLY_COMPLETION_MIN_STEPS = 100


@dataclass(frozen=True)
class AxisManualActionDeadzone:
    positive_abs: float
    negative_abs: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "positive_abs", _manual_action_deadzone_component(self.positive_abs)
        )
        object.__setattr__(
            self, "negative_abs", _manual_action_deadzone_component(self.negative_abs)
        )

    def contains(self, value: float) -> bool:
        threshold = self.positive_abs if float(value) >= 0.0 else self.negative_abs
        return abs(float(value)) <= threshold


@dataclass(frozen=True)
class ManualActionDeadzoneContract:
    axes: tuple[AxisManualActionDeadzone, ...]

    def __post_init__(self) -> None:
        if len(self.axes) != 4:
            raise ValueError("manual action deadzone contract must contain four axes")
        if not all(isinstance(axis, AxisManualActionDeadzone) for axis in self.axes):
            raise ValueError(
                "manual action deadzone contract axes must be AxisManualActionDeadzone"
            )

    def contains_action(self, action: tuple[float, float, float, float]) -> bool:
        return all(
            axis.contains(value) for axis, value in zip(self.axes, action, strict=True)
        )


@dataclass(frozen=True)
class ActSegmentSnapshot:
    """Immutable progress for the current or most recent ACT activation."""

    generation: int | None
    max_steps: int | None
    completed_steps: int
    complete: bool
    completion_reason: str | None


@dataclass(frozen=True)
class ResidentControlSnapshot:
    """One coherent control-plane view from a single core lock boundary."""

    motion: ResidentMotionSnapshot
    act_segment: ActSegmentSnapshot
    rl_is_active: bool
    act_is_active: bool
    mission_lease_active: bool
    is_operational: bool


@dataclass(frozen=True)
class _ActSegmentTracker:
    generation: int | None = None
    max_steps: int | None = None
    completed_steps: int = 0
    complete: bool = False
    final_command_seq: int | None = None
    final_action: tuple[float, float, float, float] | None = None
    handoff_due: bool = False
    last_effective_step: int | None = None
    trailing_deadzone_steps: int = 0
    completion_reason: str | None = None


@dataclass(frozen=True)
class _ActStepUpdate:
    tracker: _ActSegmentTracker
    all_axes_in_deadzone: bool


class ResidentMotionCore:
    """Own the sole STM32 command seam while policy workers remain replaceable.

    The core intentionally does not load ONNX, CUDA, cameras, ROS, or Mission
    plans.  Those workers can remain resident and submit candidates, while all
    policy selection, safety evidence, sequence allocation, schema selection,
    and serial writes stay behind this one boundary.
    """

    def __init__(
        self,
        serial_writer: SerialWriter,
        *,
        max_state_age_ms: float,
        manual_action_deadzone_contract: ManualActionDeadzoneContract | None,
        act_early_completion_chunk_steps: int = DEFAULT_ACT_ACTION_CHUNK_STEPS,
        act_early_completion_min_steps: int = DEFAULT_ACT_EARLY_COMPLETION_MIN_STEPS,
        wall_time_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        action_audit: ResidentActionAuditSink | None = None,
    ) -> None:
        self._manual_action_deadzone_contract = _manual_action_deadzone_contract(
            manual_action_deadzone_contract
        )
        self._act_early_completion_chunk_steps = _positive_integer(
            "act_early_completion_chunk_steps",
            act_early_completion_chunk_steps,
        )
        self._act_early_completion_min_steps = _positive_integer(
            "act_early_completion_min_steps",
            act_early_completion_min_steps,
        )
        self._monotonic_ns = monotonic_ns
        self._action_audit = action_audit
        self._sink = ResidentCommandSink(
            serial_writer,
            max_state_age_ms=max_state_age_ms,
            action_audit=action_audit,
        )
        self._rl = ResidentVelocityActionAdapter(
            self._sink,
            source=RL_BINDING.source,
            wall_time_ms=wall_time_ms,
            monotonic_ns=monotonic_ns,
        )
        self._act = ResidentPolicyCandidateAdapter(
            self._sink,
            binding=ACT_BINDING,
            monotonic_ns=monotonic_ns,
        )
        self._act_segment = _ActSegmentTracker()
        self._mission_lease_deadline_ns: int | None = None
        self._terminal_disarm_result: ResidentWriteResult | None = None
        self._terminal_zero_acknowledged = False
        self._control_lock = threading.RLock()

    @property
    def rl_action_sink(self) -> ResidentVelocityActionAdapter:
        """Socket-compatible sink accepted by the existing RL runners."""

        return self._rl

    @property
    def is_operational(self) -> bool:
        return self._sink.is_operational

    @property
    def rl_is_active(self) -> bool:
        return self._binding_is_active(RL_BINDING)

    @property
    def act_is_active(self) -> bool:
        return self._binding_is_active(ACT_BINDING)

    @property
    def terminal_zero_acknowledged(self) -> bool:
        with self._control_lock:
            return self._terminal_zero_acknowledged

    @property
    def mission_lease_is_active(self) -> bool:
        with self._control_lock:
            deadline = self._mission_lease_deadline_ns
            return (
                deadline is not None
                and self._sink.is_operational
                and self._monotonic_ns() < deadline
            )

    @property
    def active_act_generation(self) -> int | None:
        """Return one atomic ACT authority snapshot for state publication."""

        snapshot = self._sink.snapshot()
        if (
            snapshot.phase is HandoffPhase.ACTIVE
            and snapshot.active_binding == ACT_BINDING
        ):
            return snapshot.generation
        return None

    def initialize(self, frame: ResidentTelemetry) -> int:
        return self._sink.initialize(frame)

    def observe_telemetry(self, frame: ResidentTelemetry) -> None:
        self._sink.observe_telemetry(frame)
        with self._control_lock:
            terminal = self._terminal_disarm_result
            if terminal is not None and _terminal_zero_ack_matches(terminal, frame):
                self._terminal_zero_acknowledged = True
            segment = self._act_segment
            if not self._act_segment_acknowledged(segment, frame):
                return
            snapshot = self._sink.snapshot()
            if (
                snapshot.phase is not HandoffPhase.ACTIVE
                or snapshot.active_binding != ACT_BINDING
                or snapshot.generation != segment.generation
            ):
                return
            self._rl.begin_activation(
                now_monotonic_ns=frame.receive_monotonic_ns,
            )
            self._act_segment = replace(
                segment,
                complete=True,
                handoff_due=False,
            )

    def invalidate_telemetry(
        self,
        *,
        receive_monotonic_ns: int,
        stm32_alive: bool,
    ) -> None:
        self._sink.invalidate_telemetry(
            receive_monotonic_ns=receive_monotonic_ns,
            stm32_alive=stm32_alive,
        )

    def snapshot(self) -> ResidentMotionSnapshot:
        return self._sink.snapshot()

    def control_status_snapshot(self) -> ResidentControlSnapshot:
        """Return motion authority and ACT progress from the same instant.

        ACT completion automatically starts the ACT-to-RL handoff while holding
        ``_control_lock``.  Reading authority and segment progress separately can
        therefore combine the pre-handoff authority with post-handoff progress.
        The control wire must use this atomic view instead.
        """

        with self._control_lock:
            motion = self._sink.snapshot()
            segment = self._act_segment
            is_operational = self._sink.is_operational
            deadline = self._mission_lease_deadline_ns
            mission_lease_active = (
                deadline is not None
                and is_operational
                and self._monotonic_ns() < deadline
            )
            return ResidentControlSnapshot(
                motion=motion,
                act_segment=ActSegmentSnapshot(
                    generation=segment.generation,
                    max_steps=segment.max_steps,
                    completed_steps=segment.completed_steps,
                    complete=segment.complete,
                    completion_reason=segment.completion_reason,
                ),
                rl_is_active=(
                    motion.phase is HandoffPhase.ACTIVE
                    and motion.active_binding == RL_BINDING
                ),
                act_is_active=(
                    motion.phase is HandoffPhase.ACTIVE
                    and motion.active_binding == ACT_BINDING
                ),
                mission_lease_active=mission_lease_active,
                is_operational=is_operational,
            )

    def _binding_is_active(self, binding: PolicyBinding) -> bool:
        snapshot = self._sink.snapshot()
        return (
            snapshot.phase is HandoffPhase.ACTIVE
            and snapshot.active_binding == binding
        )

    def activate_rl(self, *, now_monotonic_ns: int | None = None) -> int:
        with self._control_lock:
            generation = self._rl.begin_activation(
                now_monotonic_ns=now_monotonic_ns
            )
            self._act_segment = replace(
                self._act_segment,
                handoff_due=False,
            )
            return generation

    def activate_act(
        self,
        *,
        max_steps: int | None = None,
        now_monotonic_ns: int | None = None,
    ) -> int:
        validated_steps = _optional_act_step_budget(max_steps)
        with self._control_lock:
            generation = self._act.begin_activation(
                now_monotonic_ns=now_monotonic_ns
            )
            current = self._act_segment
            if current.generation == generation:
                if current.max_steps != validated_steps:
                    raise ValueError(
                        "the active ACT segment budget cannot be changed"
                    )
                return generation
            self._act_segment = _ActSegmentTracker(
                generation=generation,
                max_steps=validated_steps,
            )
            return generation

    def submit_act(self, payload: bytes) -> ResidentWriteResult:
        with self._control_lock:
            if self._act_segment.handoff_due:
                reason = (
                    "act_segment_early_complete"
                    if self._act_segment.completion_reason == "deadzone_chunk"
                    else "act_segment_budget_reached"
                )
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason=reason,
                    command_seq=None,
                    mode=ControlMode.MANUAL_ACTION,
                    effective_action=ZERO_ACTION,
                )
            result = self._act.send(payload)
            segment = self._act_segment
            if not result.accepted or segment.generation is None:
                return result
            candidate = decode_motion_candidate(payload)
            update = _advance_act_segment(
                segment,
                result,
                action_chunk=candidate.action_chunk,
                manual_action_deadzone_contract=self._manual_action_deadzone_contract,
                early_completion_chunk_steps=(
                    self._act_early_completion_chunk_steps
                ),
                early_completion_min_steps=(
                    self._act_early_completion_min_steps
                ),
            )
            self._act_segment = update.tracker
            self._audit_act_step(update, result)
            return result

    def _audit_act_step(
        self,
        update: _ActStepUpdate,
        result: ResidentWriteResult,
    ) -> None:
        segment = update.tracker
        common = {
            "runtime_id": self._sink.runtime_id,
            "generation": segment.generation,
            "completed_steps": segment.completed_steps,
            "max_steps": segment.max_steps,
            "manual_deadzone_contract_enabled": (
                self._manual_action_deadzone_contract is not None
            ),
        }
        emit_action_audit(
            self._action_audit,
            "act_step",
            monotonic_ns=self._monotonic_ns(),
            command_seq=result.command_seq,
            action=list(result.effective_action),
            all_axes_in_deadzone=update.all_axes_in_deadzone,
            **common,
        )
        if segment.handoff_due:
            emit_action_audit(
                self._action_audit,
                "act_segment_summary",
                monotonic_ns=self._monotonic_ns(),
                last_effective_step=segment.last_effective_step,
                trailing_deadzone_steps=segment.trailing_deadzone_steps,
                estimated_trailing_deadzone_ms=(
                    segment.trailing_deadzone_steps * ACT_NOMINAL_STEP_PERIOD_MS
                ),
                completion_reason=segment.completion_reason,
                deadzone_chunk_steps=self._act_early_completion_chunk_steps,
                skipped_budget_steps=(
                    max(0, segment.max_steps - segment.completed_steps)
                    if segment.max_steps is not None
                    else 0
                ),
                **common,
            )

    def act_segment_snapshot(self) -> ActSegmentSnapshot:
        with self._control_lock:
            segment = self._act_segment
            return ActSegmentSnapshot(
                generation=segment.generation,
                max_steps=segment.max_steps,
                completed_steps=segment.completed_steps,
                complete=segment.complete,
                completion_reason=segment.completion_reason,
            )

    def notify_act_worker_disconnected(
        self,
        *,
        now_monotonic_ns: int | None = None,
    ) -> bool:
        """Revoke the current ACT generation when its local worker disappears.

        The Unix data link is not a motion authority.  Losing it therefore
        revokes the ACT generation at this sole serial owner and emits the
        matching manual-action zero without waiting for the STM32 watchdog.
        """

        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._control_lock:
            if not self._sink.is_operational:
                return False
            generation = self._act_segment.generation
            if generation is None:
                return False
            revoked = self._sink.request_source_stop(
                ACT_BINDING,
                generation=generation,
                now_monotonic_ns=now,
            )
            if revoked:
                self._act_segment = replace(
                    self._act_segment,
                    handoff_due=False,
                )
            return revoked

    def renew_mission_lease(
        self,
        *,
        lease_ms: int,
        now_monotonic_ns: int | None = None,
    ) -> None:
        """Renew the PC Mission's bounded authority to keep this owner armed."""

        if (
            isinstance(lease_ms, bool)
            or not isinstance(lease_ms, int)
            or not MIN_MISSION_LEASE_MS <= lease_ms <= MAX_MISSION_LEASE_MS
        ):
            raise ValueError(
                "lease_ms must be an integer within "
                f"[{MIN_MISSION_LEASE_MS}, {MAX_MISSION_LEASE_MS}]"
            )
        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._control_lock:
            if not self._sink.is_operational:
                raise RuntimeError("resident motion core is terminally disarmed")
            deadline = self._mission_lease_deadline_ns
            if deadline is not None and now >= deadline:
                self.terminal_disarm(now_monotonic_ns=now)
                raise RuntimeError(
                    "resident Mission lease expired; motion core terminally disarmed"
                )
            self._mission_lease_deadline_ns = now + lease_ms * 1_000_000

    def tick(
        self,
        *,
        now_monotonic_ns: int | None = None,
    ) -> ResidentWriteResult | None:
        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._control_lock:
            deadline = self._mission_lease_deadline_ns
            if deadline is not None and now >= deadline:
                self._mission_lease_deadline_ns = None
                return self.terminal_disarm(now_monotonic_ns=now)
            return self._sink.tick(now_monotonic_ns=now)

    def terminal_disarm(
        self,
        *,
        now_monotonic_ns: int | None = None,
    ) -> ResidentWriteResult:
        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._control_lock:
            if self._terminal_disarm_result is not None:
                return self._terminal_disarm_result
            self._mission_lease_deadline_ns = None
            self._act_segment = replace(
                self._act_segment,
                handoff_due=False,
            )
            result = self._sink.terminal_disarm(now_monotonic_ns=now)
            self._terminal_disarm_result = result
            self._terminal_zero_acknowledged = (
                not result.write_performed
                and result.reason == "terminal_disarm"
                and result.command_seq is None
                and result.mode is None
                and result.effective_action == ZERO_ACTION
            )
            return result

    @staticmethod
    def _act_segment_acknowledged(
        segment: _ActSegmentTracker,
        frame: ResidentTelemetry,
    ) -> bool:
        return (
            segment.handoff_due
            and not segment.complete
            and segment.final_command_seq is not None
            and segment.final_action is not None
            and frame.command_rx_seq == segment.final_command_seq
            and frame.command_valid
            and not frame.command_timed_out
            and frame.control_mode is ControlMode.MANUAL_ACTION
        )


def _advance_act_segment(
    segment: _ActSegmentTracker,
    result: ResidentWriteResult,
    *,
    action_chunk: tuple[tuple[float, float, float, float], ...] | None,
    manual_action_deadzone_contract: ManualActionDeadzoneContract | None,
    early_completion_chunk_steps: int,
    early_completion_min_steps: int,
) -> _ActStepUpdate:
    completed_steps = segment.completed_steps + 1
    all_axes_in_deadzone = _action_within_manual_deadzone_contract(
        result.effective_action,
        manual_action_deadzone_contract,
    )
    trailing_deadzone_steps = (
        segment.trailing_deadzone_steps + 1 if all_axes_in_deadzone else 0
    )
    budget_due = (
        segment.max_steps is not None and completed_steps >= segment.max_steps
    )
    deadzone_chunk_due = (
        manual_action_deadzone_contract is not None
        and segment.max_steps is not None
        and completed_steps < segment.max_steps
        and segment.completed_steps >= early_completion_min_steps
        and action_chunk is not None
        and len(action_chunk) == early_completion_chunk_steps
        and all(
            _action_within_manual_deadzone_contract(
                action,
                manual_action_deadzone_contract,
            )
            for action in action_chunk
        )
    )
    handoff_due = budget_due or deadzone_chunk_due
    completion_reason = (
        "step_budget"
        if budget_due
        else "deadzone_chunk" if deadzone_chunk_due else None
    )
    return _ActStepUpdate(
        tracker=replace(
            segment,
            completed_steps=completed_steps,
            final_command_seq=result.command_seq if handoff_due else None,
            final_action=result.effective_action if handoff_due else None,
            handoff_due=handoff_due,
            last_effective_step=(
                segment.last_effective_step
                if all_axes_in_deadzone
                else completed_steps
            ),
            trailing_deadzone_steps=trailing_deadzone_steps,
            completion_reason=completion_reason,
        ),
        all_axes_in_deadzone=all_axes_in_deadzone,
    )


def _manual_action_deadzone_component(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) < 1.0
    ):
        raise ValueError(
            "manual action deadzone components must be finite and in [0, 1)"
        )
    return float(value)


def _manual_action_deadzone_contract(
    value: ManualActionDeadzoneContract | None,
) -> ManualActionDeadzoneContract | None:
    if value is None:
        return None
    if not isinstance(value, ManualActionDeadzoneContract):
        raise ValueError(
            "manual_action_deadzone_contract must be a ManualActionDeadzoneContract or None"
        )
    return value


def _action_within_manual_deadzone_contract(
    action: tuple[float, float, float, float],
    contract: ManualActionDeadzoneContract | None,
) -> bool:
    if contract is None:
        return False
    return contract.contains_action(action)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_act_step_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_ACT_SEGMENT_STEPS
    ):
        raise ValueError(
            f"max_steps must be an integer in [1, {MAX_ACT_SEGMENT_STEPS}]"
        )
    return value


def _terminal_zero_ack_matches(
    result: ResidentWriteResult,
    frame: ResidentTelemetry,
) -> bool:
    return (
        result.write_performed
        and result.command_seq is not None
        and result.mode is not None
        and result.effective_action == ZERO_ACTION
        and frame.command_rx_seq == result.command_seq
        and frame.command_valid
        and not frame.command_timed_out
        and frame.control_mode is result.mode
        and frame.command_action == ZERO_ACTION
    )
