"""Orin-local declarative Mission orchestration for V3-B.

This Module owns only deterministic phase sequencing and immutable trajectory
references.  Policy inference, trajectory tracking, motion authority, serial
writes, and terminal-zero acknowledgement remain behind their existing
Interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from ._fixed_cycle_plan import (
    CATALOG_SCHEMA_VERSION,
    MAX_REQUESTED_CYCLES,
    SCHEMA_VERSION,
    FixedCyclePlan,
    FixedTargetCatalogArtifact,
    FixedTrajectoryArtifact,
    FixedTrajectoryTemplate,
    _identifier,
    _integer,
    load_fixed_cycle_plan,
    load_fixed_cycle_registry,
    verify_fixed_cycle_artifacts,
)
from ._resident_mission_definition import MissionBehavior


@dataclass(frozen=True)
class FixedCycleDirective:
    stage: str
    child: str
    target_id: str | None = None
    trajectory: FixedTrajectoryArtifact | None = None
    max_steps: int | None = None
    behavior: str | None = None
    behavior_id: str = ""


@dataclass(frozen=True)
class FixedCycleSnapshot:
    run_id: str = ""
    stage: str = "IDLE"
    requested_cycles: int = 0
    completed_cycles: int = 0
    current_dig_point_id: str = ""
    dig_group_id: str = ""
    terminal: bool = False
    outcome: str = ""
    reason_code: str = ""
    mission_id: str = ""
    active_behavior_id: str = ""


class ResidentFixedCycle:
    """Pure deterministic state machine for one local multi-cycle Mission."""

    def __init__(self, plan: FixedCyclePlan) -> None:
        if not isinstance(plan, FixedCyclePlan):
            raise ValueError("plan must be a FixedCyclePlan")
        self._plan = plan
        self._snapshot = FixedCycleSnapshot()
        self._dig_sequence_index = 0
        self._active_dig_sequence = plan.dig_sequence
        self._active_behavior = plan.mission.entry_behavior
        self._cycle_behavior_index = -1

    @property
    def snapshot(self) -> FixedCycleSnapshot:
        return self._snapshot

    def start(
        self,
        *,
        run_id: str,
        requested_cycles: int,
        first_dig_point_id: str | None = None,
        dig_group_id: str | None = None,
    ) -> FixedCycleDirective:
        if self._snapshot.stage != "IDLE" and not self._snapshot.terminal:
            raise RuntimeError("a resident fixed cycle is already active")
        cycles = _integer(
            "requested_cycles",
            requested_cycles,
            1,
            MAX_REQUESTED_CYCLES,
        )
        group_id = dig_group_id or self._plan.default_dig_group
        if group_id not in self._plan.dig_groups:
            raise ValueError("dig_group_id is not in the fixed plan")
        self._active_dig_sequence = self._plan.dig_groups[group_id]
        first = first_dig_point_id or self._active_dig_sequence[0]
        if first not in self._active_dig_sequence:
            raise ValueError("first_dig_point_id is not in the selected dig group")
        self._dig_sequence_index = self._active_dig_sequence.index(first)
        self._snapshot = FixedCycleSnapshot(
            run_id=_identifier("run_id", run_id),
            stage=self._plan.mission.entry_behavior.stage_id,
            requested_cycles=cycles,
            current_dig_point_id=first,
            dig_group_id=group_id,
            mission_id=self._plan.mission.mission_id,
            active_behavior_id=self._plan.mission.entry_behavior.behavior_id,
        )
        self._active_behavior = self._plan.mission.entry_behavior
        self._cycle_behavior_index = -1
        return self._directive_for(self._active_behavior)

    def record_child_result(
        self,
        *,
        child: str,
        outcome: str,
        reason_code: str,
        quiescence_confirmed: bool,
        completed_steps: int | None = None,
    ) -> FixedCycleDirective | None:
        if self._snapshot.stage == "IDLE" or self._snapshot.terminal:
            raise RuntimeError("no resident fixed cycle child is active")
        if quiescence_confirmed is not True:
            self._fail("CHILD_NOT_QUIESCENT")
            return None
        if outcome == "CANCELLED":
            self._cancel(reason_code or "CANCELLED")
            return None
        if outcome != "SUCCEEDED":
            self._fail(reason_code or "CHILD_FAILED")
            return None

        behavior = self._active_behavior
        if child != behavior.contract.adapter:
            self._fail("UNEXPECTED_CHILD_RESULT")
            return None
        if not self._successful_result(
            behavior,
            reason_code=reason_code,
            completed_steps=completed_steps,
        ):
            return None
        if self._cycle_behavior_index < 0:
            return self._activate_cycle_behavior(0)
        if self._cycle_behavior_index == len(self._plan.mission.cycle_behaviors) - 1:
            if self._snapshot.completed_cycles == self._snapshot.requested_cycles:
                self._snapshot = replace(
                    self._snapshot,
                    stage="COMPLETED",
                    active_behavior_id="",
                    terminal=True,
                    outcome="SUCCEEDED",
                    reason_code="SEQUENCE_COMPLETED",
                )
                return None
            return self._activate_cycle_behavior(0)
        return self._activate_cycle_behavior(self._cycle_behavior_index + 1)

    def cancel(self, *, reason_code: str = "CANCELLED") -> None:
        if self._snapshot.stage == "IDLE" or self._snapshot.terminal:
            raise RuntimeError("no active resident fixed cycle can be cancelled")
        self._cancel(reason_code)

    def fail(self, *, reason_code: str) -> None:
        """Publish a terminal failure raised by a local Runtime boundary."""

        if self._snapshot.stage == "IDLE" or self._snapshot.terminal:
            raise RuntimeError("no active resident fixed cycle can be failed")
        self._fail(_identifier("reason_code", reason_code))

    def _successful_result(
        self,
        behavior: MissionBehavior,
        *,
        reason_code: str,
        completed_steps: int | None,
    ) -> bool:
        if behavior.contract.adapter == "follow":
            if reason_code == "SUCCEEDED":
                return True
            self._fail("UNEXPECTED_CHILD_RESULT")
            return False
        if behavior.contract.adapter == "fixed_action":
            if reason_code == "SEQUENCE_COMPLETED":
                return True
            self._fail("UNEXPECTED_CHILD_RESULT")
            return False
        assert behavior.max_steps is not None
        budget_reached = (
            reason_code == "STEP_BUDGET_REACHED"
            and completed_steps == behavior.max_steps
        )
        early_completion = (
            behavior.contract.allow_deadzone_early_completion
            and reason_code == "DEADZONE_CHUNK_REACHED"
            and isinstance(completed_steps, int)
            and not isinstance(completed_steps, bool)
            and 0 < completed_steps < behavior.max_steps
        )
        if budget_reached or early_completion:
            return True
        self._fail("ACT_STEP_BUDGET_MISMATCH")
        return False

    def _activate_cycle_behavior(self, index: int) -> FixedCycleDirective:
        behavior = self._plan.mission.cycle_behaviors[index]
        if behavior.target_role == "return_dig":
            completed = self._snapshot.completed_cycles + 1
            if completed < self._snapshot.requested_cycles:
                self._dig_sequence_index = (
                    self._dig_sequence_index + 1
                ) % len(self._active_dig_sequence)
            self._snapshot = replace(
                self._snapshot,
                completed_cycles=completed,
                current_dig_point_id=self._active_dig_sequence[
                    self._dig_sequence_index
                ],
            )
        self._cycle_behavior_index = index
        self._active_behavior = behavior
        self._snapshot = replace(
            self._snapshot,
            stage=behavior.stage_id,
            active_behavior_id=behavior.behavior_id,
        )
        return self._directive_for(behavior)

    def _directive_for(self, behavior: MissionBehavior) -> FixedCycleDirective:
        contract = behavior.contract
        if contract.adapter == "follow":
            target_id = (
                "dump"
                if behavior.target_role == "dump"
                else self._snapshot.current_dig_point_id
            )
            return FixedCycleDirective(
                stage=behavior.stage_id,
                child="follow",
                target_id=target_id,
                trajectory=self._plan.trajectories[target_id],
                behavior_id=behavior.behavior_id,
            )
        if contract.adapter == "act":
            return FixedCycleDirective(
                stage=behavior.stage_id,
                child="act",
                max_steps=behavior.max_steps,
                behavior_id=behavior.behavior_id,
            )
        assert contract.fixed_action is not None
        return FixedCycleDirective(
            stage=behavior.stage_id,
            child="fixed_action",
            behavior=contract.fixed_action,
            behavior_id=behavior.behavior_id,
        )

    def _cancel(self, reason_code: str) -> None:
        self._snapshot = replace(
            self._snapshot,
            stage="CANCELLED",
            active_behavior_id="",
            terminal=True,
            outcome="CANCELLED",
            reason_code=reason_code,
        )

    def _fail(self, reason_code: str) -> None:
        self._snapshot = replace(
            self._snapshot,
            stage="FAILED",
            active_behavior_id="",
            terminal=True,
            outcome="FAILED",
            reason_code=reason_code,
        )


class ResidentFixedCycleDriver(Protocol):
    def start_follow(self, artifact: FixedTrajectoryArtifact) -> None: ...

    def activate_act(self, *, max_steps: int) -> None: ...

    def start_fixed_action(self, behavior: str) -> None: ...

    def terminal_disarm(self) -> None: ...


class ResidentFixedCycleCoordinator:
    """Dispatch state-machine directives to local resident Runtime Interfaces."""

    def __init__(self, *, plan: FixedCyclePlan, driver: ResidentFixedCycleDriver) -> None:
        self._cycle = ResidentFixedCycle(plan)
        self._driver = driver
        self._terminal_disarmed = False

    @property
    def snapshot(self) -> FixedCycleSnapshot:
        return self._cycle.snapshot

    def start(
        self,
        *,
        run_id: str,
        requested_cycles: int,
        first_dig_point_id: str | None = None,
        dig_group_id: str | None = None,
    ) -> None:
        self._terminal_disarmed = False
        directive = self._cycle.start(
            run_id=run_id,
            requested_cycles=requested_cycles,
            first_dig_point_id=first_dig_point_id,
            dig_group_id=dig_group_id,
        )
        self._dispatch(directive)

    def record_child_result(self, **result: Any) -> None:
        directive = self._cycle.record_child_result(**result)
        if directive is not None:
            self._dispatch(directive)
            return
        if self._cycle.snapshot.terminal:
            self._terminal_disarm()

    def cancel(self) -> None:
        self._terminal_disarm()
        self._cycle.cancel()

    def fail(self, *, reason_code: str) -> None:
        """Fail the active cycle after first closing the sole motion owner."""

        self._terminal_disarm()
        self._cycle.fail(reason_code=reason_code)

    def _dispatch(self, directive: FixedCycleDirective) -> None:
        try:
            self._dispatch_unchecked(directive)
        except Exception:
            self._cycle.fail(reason_code="LOCAL_DISPATCH_FAILED")
            try:
                self._terminal_disarm()
            except Exception:
                raise RuntimeError(
                    "resident fixed cycle local dispatch and terminal disarm failed"
                ) from None
            raise RuntimeError("resident fixed cycle local dispatch failed") from None

    def _dispatch_unchecked(self, directive: FixedCycleDirective) -> None:
        if directive.child == "follow":
            assert directive.trajectory is not None
            self._driver.start_follow(directive.trajectory)
            return
        if directive.child == "act":
            assert directive.max_steps is not None
            self._driver.activate_act(max_steps=directive.max_steps)
            return
        if directive.child == "fixed_action":
            assert directive.behavior is not None
            self._driver.start_fixed_action(directive.behavior)
            return
        raise RuntimeError("resident fixed cycle emitted an unsupported directive")

    def _terminal_disarm(self) -> None:
        if self._terminal_disarmed:
            return
        self._driver.terminal_disarm()
        self._terminal_disarmed = True
