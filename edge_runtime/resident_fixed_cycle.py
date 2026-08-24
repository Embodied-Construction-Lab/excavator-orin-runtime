"""Orin-local fixed-target hybrid cycle orchestration for V3-A.

This Module owns only deterministic phase sequencing and immutable trajectory
references.  Policy inference, trajectory tracking, motion authority, serial
writes, and terminal-zero acknowledgement remain behind their existing
Interfaces.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol


SCHEMA_VERSION = "resident_fixed_cycle_plan.v1"
MAX_REQUESTED_CYCLES = 9
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "plan_id",
        "validation_status",
        "dig_sequence",
        "act_max_steps",
        "trajectories",
    }
)
_ARTIFACT_FIELDS = frozenset({"trajectory_id", "phase", "path", "sha256"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FixedTrajectoryArtifact:
    """One immutable, field-validated trajectory deployed on the Orin."""

    trajectory_id: str
    phase: str
    path: Path
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        expected_phase: str,
    ) -> "FixedTrajectoryArtifact":
        if not isinstance(value, Mapping) or set(value) != _ARTIFACT_FIELDS:
            raise ValueError("fixed trajectory artifact fields are invalid")
        phase = _text("trajectory phase", value["phase"])
        if phase != expected_phase:
            raise ValueError("fixed trajectory phase does not match target")
        path = Path(_text("trajectory path", value["path"]))
        if not path.is_absolute():
            raise ValueError("fixed trajectory path must be absolute")
        digest = value["sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("fixed trajectory sha256 must be lowercase hexadecimal")
        return cls(
            trajectory_id=_identifier("trajectory_id", value["trajectory_id"]),
            phase=phase,
            path=path,
            sha256=digest,
        )


@dataclass(frozen=True)
class FixedCyclePlan:
    """Strict fixed-target plan loaded before any motion resource is opened."""

    plan_id: str
    validation_status: str
    dig_sequence: tuple[str, ...]
    act_max_steps: int
    trajectories: Mapping[str, FixedTrajectoryArtifact]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixedCyclePlan":
        if not isinstance(value, Mapping) or set(value) != _PLAN_FIELDS:
            raise ValueError("resident fixed cycle plan fields are invalid")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported resident fixed cycle plan schema")
        if value["validation_status"] != "field_validated":
            raise ValueError("resident fixed cycle plan must be field_validated")
        raw_sequence = value["dig_sequence"]
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise ValueError("dig_sequence must be a non-empty list")
        sequence = tuple(_identifier("dig point id", item) for item in raw_sequence)
        if len(set(sequence)) != len(sequence):
            raise ValueError("dig_sequence must contain unique target ids")
        act_max_steps = _integer("act_max_steps", value["act_max_steps"], 1, 2000)
        raw_trajectories = value["trajectories"]
        expected_targets = frozenset((*sequence, "dump"))
        if (
            not isinstance(raw_trajectories, Mapping)
            or set(raw_trajectories) != expected_targets
        ):
            raise ValueError(
                "trajectory targets must exactly match dig_sequence plus dump"
            )
        artifacts = {
            target_id: FixedTrajectoryArtifact.from_mapping(
                raw_trajectories[target_id],
                expected_phase="dump" if target_id == "dump" else "dig",
            )
            for target_id in (*sequence, "dump")
        }
        return cls(
            plan_id=_identifier("plan_id", value["plan_id"]),
            validation_status="field_validated",
            dig_sequence=sequence,
            act_max_steps=act_max_steps,
            trajectories=MappingProxyType(artifacts),
        )


def load_fixed_cycle_plan(path: str | Path) -> FixedCyclePlan:
    """Load one strict plan without opening policy, camera, or serial resources."""

    plan_path = Path(path)
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    return FixedCyclePlan.from_mapping(document)


def verify_fixed_cycle_artifacts(
    plan: FixedCyclePlan,
) -> Mapping[str, bytes]:
    """Load and hash every fixed trajectory before motion resources are opened.

    Returning the verified bytes lets the eventual Runtime materialize its
    in-memory registry without reopening mutable paths after this boundary.
    """

    if not isinstance(plan, FixedCyclePlan):
        raise ValueError("plan must be a FixedCyclePlan")
    verified: dict[str, bytes] = {}
    for target_id in (*plan.dig_sequence, "dump"):
        artifact = plan.trajectories[target_id]
        payload = _read_regular_file_no_follow(artifact.path)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(
                f"fixed trajectory {target_id} sha256 mismatch"
            )
        verified[target_id] = payload
    return MappingProxyType(verified)


@dataclass(frozen=True)
class FixedCycleDirective:
    stage: str
    child: str
    target_id: str | None = None
    trajectory: FixedTrajectoryArtifact | None = None
    max_steps: int | None = None
    behavior: str | None = None


@dataclass(frozen=True)
class FixedCycleSnapshot:
    run_id: str = ""
    stage: str = "IDLE"
    requested_cycles: int = 0
    completed_cycles: int = 0
    current_dig_point_id: str = ""
    terminal: bool = False
    outcome: str = ""
    reason_code: str = ""


class ResidentFixedCycle:
    """Pure deterministic state machine for one local multi-cycle Mission."""

    def __init__(self, plan: FixedCyclePlan) -> None:
        if not isinstance(plan, FixedCyclePlan):
            raise ValueError("plan must be a FixedCyclePlan")
        self._plan = plan
        self._snapshot = FixedCycleSnapshot()
        self._dig_sequence_index = 0

    @property
    def snapshot(self) -> FixedCycleSnapshot:
        return self._snapshot

    def start(
        self,
        *,
        run_id: str,
        requested_cycles: int,
        first_dig_point_id: str | None = None,
    ) -> FixedCycleDirective:
        if self._snapshot.stage != "IDLE" and not self._snapshot.terminal:
            raise RuntimeError("a resident fixed cycle is already active")
        cycles = _integer(
            "requested_cycles",
            requested_cycles,
            1,
            MAX_REQUESTED_CYCLES,
        )
        first = first_dig_point_id or self._plan.dig_sequence[0]
        if first not in self._plan.dig_sequence:
            raise ValueError("first_dig_point_id is not in the fixed plan")
        self._dig_sequence_index = self._plan.dig_sequence.index(first)
        self._snapshot = FixedCycleSnapshot(
            run_id=_identifier("run_id", run_id),
            stage="FOLLOW_DIG",
            requested_cycles=cycles,
            current_dig_point_id=first,
        )
        return self._follow_directive(target_id=first, stage="FOLLOW_DIG")

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

        stage = self._snapshot.stage
        if stage == "FOLLOW_DIG" and child == "follow" and reason_code == "SUCCEEDED":
            self._snapshot = replace(self._snapshot, stage="ACT_DIG")
            return FixedCycleDirective(
                stage="ACT_DIG",
                child="act",
                max_steps=self._plan.act_max_steps,
            )
        if stage == "ACT_DIG" and child == "act":
            if (
                reason_code != "STEP_BUDGET_REACHED"
                or completed_steps != self._plan.act_max_steps
            ):
                self._fail("ACT_STEP_BUDGET_MISMATCH")
                return None
            self._snapshot = replace(self._snapshot, stage="FOLLOW_DUMP")
            return self._follow_directive(target_id="dump", stage="FOLLOW_DUMP")
        if stage == "FOLLOW_DUMP" and child == "follow" and reason_code == "SUCCEEDED":
            self._snapshot = replace(self._snapshot, stage="EXECUTE_DUMP")
            return FixedCycleDirective(
                stage="EXECUTE_DUMP",
                child="fixed_action",
                behavior="ExecuteDump",
            )
        if (
            stage == "EXECUTE_DUMP"
            and child == "fixed_action"
            and reason_code == "SEQUENCE_COMPLETED"
        ):
            return self._complete_or_start_next_dig()
        self._fail("UNEXPECTED_CHILD_RESULT")
        return None

    def cancel(self, *, reason_code: str = "CANCELLED") -> None:
        if self._snapshot.stage == "IDLE" or self._snapshot.terminal:
            raise RuntimeError("no active resident fixed cycle can be cancelled")
        self._cancel(reason_code)

    def fail(self, *, reason_code: str) -> None:
        """Publish a terminal failure raised by a local Runtime boundary."""

        if self._snapshot.stage == "IDLE" or self._snapshot.terminal:
            raise RuntimeError("no active resident fixed cycle can be failed")
        self._fail(_identifier("reason_code", reason_code))

    def _complete_or_start_next_dig(self) -> FixedCycleDirective | None:
        completed = self._snapshot.completed_cycles + 1
        if completed >= self._snapshot.requested_cycles:
            self._snapshot = replace(
                self._snapshot,
                stage="COMPLETED",
                completed_cycles=completed,
                terminal=True,
                outcome="SUCCEEDED",
                reason_code="SEQUENCE_COMPLETED",
            )
            return None
        self._dig_sequence_index = (
            self._dig_sequence_index + 1
        ) % len(self._plan.dig_sequence)
        target_id = self._plan.dig_sequence[self._dig_sequence_index]
        self._snapshot = replace(
            self._snapshot,
            stage="FOLLOW_DIG",
            completed_cycles=completed,
            current_dig_point_id=target_id,
        )
        return self._follow_directive(target_id=target_id, stage="FOLLOW_DIG")

    def _follow_directive(
        self,
        *,
        target_id: str,
        stage: str,
    ) -> FixedCycleDirective:
        return FixedCycleDirective(
            stage=stage,
            child="follow",
            target_id=target_id,
            trajectory=self._plan.trajectories[target_id],
        )

    def _cancel(self, reason_code: str) -> None:
        self._snapshot = replace(
            self._snapshot,
            stage="CANCELLED",
            terminal=True,
            outcome="CANCELLED",
            reason_code=reason_code,
        )

    def _fail(self, reason_code: str) -> None:
        self._snapshot = replace(
            self._snapshot,
            stage="FAILED",
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
    ) -> None:
        self._terminal_disarmed = False
        directive = self._cycle.start(
            run_id=run_id,
            requested_cycles=requested_cycles,
            first_dig_point_id=first_dig_point_id,
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


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe non-empty identifier")
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _integer(name: str, value: Any, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _read_regular_file_no_follow(path: Path) -> bytes:
    try:
        path_status = path.lstat()
    except OSError as exc:
        raise ValueError(f"cannot read fixed trajectory artifact: {exc}") from exc
    if not stat.S_ISREG(path_status.st_mode):
        raise ValueError(
            "fixed trajectory artifact must be a regular non-symbolic-link file"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or opened_status.st_dev != path_status.st_dev
            or opened_status.st_ino != path_status.st_ino
        ):
            raise ValueError("fixed trajectory artifact changed during open")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read fixed trajectory artifact: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
