"""One resident motion owner shared by RL and imitation-learning workers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Callable

from .resident_ingress import (
    ResidentPolicyCandidateAdapter,
    ResidentVelocityActionAdapter,
)
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


@dataclass(frozen=True)
class ActSegmentSnapshot:
    """Immutable progress for the current or most recent ACT activation."""

    generation: int | None
    max_steps: int | None
    completed_steps: int
    complete: bool


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
        wall_time_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._monotonic_ns = monotonic_ns
        self._sink = ResidentCommandSink(
            serial_writer,
            max_state_age_ms=max_state_age_ms,
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
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason="act_segment_budget_reached",
                    command_seq=None,
                    mode=ControlMode.MANUAL_ACTION,
                    effective_action=ZERO_ACTION,
                )
            result = self._act.send(payload)
            segment = self._act_segment
            if not result.accepted or segment.generation is None:
                return result
            completed_steps = segment.completed_steps + 1
            handoff_due = (
                segment.max_steps is not None
                and completed_steps >= segment.max_steps
            )
            self._act_segment = replace(
                segment,
                completed_steps=completed_steps,
                final_command_seq=(
                    result.command_seq if handoff_due else None
                ),
                final_action=(
                    result.effective_action if handoff_due else None
                ),
                handoff_due=handoff_due,
            )
            return result

    def act_segment_snapshot(self) -> ActSegmentSnapshot:
        with self._control_lock:
            segment = self._act_segment
            return ActSegmentSnapshot(
                generation=segment.generation,
                max_steps=segment.max_steps,
                completed_steps=segment.completed_steps,
                complete=segment.complete,
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
