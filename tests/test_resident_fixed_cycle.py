import hashlib
import json
from pathlib import Path

import pytest

from edge_runtime.resident_fixed_cycle import (
    FixedCyclePlan,
    ResidentFixedCycle,
    ResidentFixedCycleCoordinator,
    load_fixed_cycle_plan,
    verify_fixed_cycle_artifacts,
)


def _artifact(target_id: str, phase: str) -> dict[str, object]:
    return {
        "trajectory_id": f"field-{target_id}-v1",
        "phase": phase,
        "path": f"/opt/excavator-trajectories/{target_id}.json",
        "sha256": ("a" if phase == "dig" else "b") * 64,
    }


def _plan_document() -> dict[str, object]:
    return {
        "schema_version": "resident_fixed_cycle_plan.v1",
        "plan_id": "field-cycle-v3a",
        "validation_status": "field_validated",
        "dig_sequence": ["dig_01", "dig_02", "dig_03"],
        "act_max_steps": 130,
        "trajectories": {
            "dig_01": _artifact("dig_01", "dig"),
            "dig_02": _artifact("dig_02", "dig"),
            "dig_03": _artifact("dig_03", "dig"),
            "dump": _artifact("dump", "dump"),
        },
    }


def _plan() -> FixedCyclePlan:
    return FixedCyclePlan.from_mapping(_plan_document())


def test_plan_loader_is_strict_and_requires_field_validated_artifacts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resident-cycle.json"
    path.write_text(json.dumps(_plan_document()), encoding="utf-8")

    plan = load_fixed_cycle_plan(path)

    assert plan.plan_id == "field-cycle-v3a"
    assert plan.dig_sequence == ("dig_01", "dig_02", "dig_03")
    assert plan.act_max_steps == 130
    assert plan.trajectories["dump"].phase == "dump"

    invalid = _plan_document()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        FixedCyclePlan.from_mapping(invalid)

    unvalidated = _plan_document()
    unvalidated["validation_status"] = "uncommissioned"
    with pytest.raises(ValueError, match="field_validated"):
        FixedCyclePlan.from_mapping(unvalidated)


def test_plan_rejects_missing_target_hash_drift_and_non_absolute_paths() -> None:
    missing = _plan_document()
    del missing["trajectories"]["dig_03"]
    with pytest.raises(ValueError, match="exactly match"):
        FixedCyclePlan.from_mapping(missing)

    bad_hash = _plan_document()
    bad_hash["trajectories"]["dump"]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="sha256"):
        FixedCyclePlan.from_mapping(bad_hash)

    relative = _plan_document()
    relative["trajectories"]["dig_01"]["path"] = "dig_01.json"
    with pytest.raises(ValueError, match="absolute"):
        FixedCyclePlan.from_mapping(relative)


def test_artifact_preflight_hashes_every_trajectory_before_motion(tmp_path: Path) -> None:
    document = _plan_document()
    for target_id, artifact in document["trajectories"].items():
        path = tmp_path / f"{target_id}.json"
        payload = json.dumps({"target_id": target_id}).encode("utf-8")
        path.write_bytes(payload)
        artifact["path"] = str(path)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    plan = FixedCyclePlan.from_mapping(document)

    verify_fixed_cycle_artifacts(plan)

    Path(plan.trajectories["dump"].path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_fixed_cycle_artifacts(plan)


def test_artifact_preflight_rejects_symbolic_links(tmp_path: Path) -> None:
    document = _plan_document()
    for target_id, artifact in document["trajectories"].items():
        path = tmp_path / f"{target_id}.json"
        path.write_text(target_id, encoding="utf-8")
        artifact["path"] = str(path)
        artifact["sha256"] = hashlib.sha256(target_id.encode()).hexdigest()
    real = tmp_path / "dig_01.json"
    link = tmp_path / "dig_01-link.json"
    link.symlink_to(real)
    document["trajectories"]["dig_01"]["path"] = str(link)
    plan = FixedCyclePlan.from_mapping(document)

    with pytest.raises(ValueError, match="regular non-symbolic-link"):
        verify_fixed_cycle_artifacts(plan)


def test_three_cycles_advance_locally_without_per_stage_external_commands() -> None:
    cycle = ResidentFixedCycle(_plan())

    directive = cycle.start(
        run_id="run-001",
        requested_cycles=3,
        first_dig_point_id="dig_01",
    )
    observed = [(directive.stage, directive.target_id)]

    for expected_dig in ("dig_01", "dig_02", "dig_03"):
        assert directive.stage == "FOLLOW_DIG"
        assert directive.target_id == expected_dig
        directive = cycle.record_child_result(
            child="follow",
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        observed.append((directive.stage, directive.max_steps))
        assert directive.stage == "ACT_DIG"
        assert directive.max_steps == 130

        directive = cycle.record_child_result(
            child="act",
            outcome="SUCCEEDED",
            reason_code="STEP_BUDGET_REACHED",
            quiescence_confirmed=True,
            completed_steps=130,
        )
        observed.append((directive.stage, directive.target_id))
        assert directive.stage == "FOLLOW_DUMP"
        assert directive.target_id == "dump"

        directive = cycle.record_child_result(
            child="follow",
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        observed.append((directive.stage, directive.behavior))
        assert directive.stage == "EXECUTE_DUMP"
        assert directive.behavior == "ExecuteDump"

        directive = cycle.record_child_result(
            child="fixed_action",
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )
        if directive is not None:
            observed.append((directive.stage, directive.target_id))

    assert directive is None
    assert cycle.snapshot.stage == "COMPLETED"
    assert cycle.snapshot.completed_cycles == 3
    assert cycle.snapshot.outcome == "SUCCEEDED"
    assert [item for item in observed if item[0] == "FOLLOW_DIG"] == [
        ("FOLLOW_DIG", "dig_01"),
        ("FOLLOW_DIG", "dig_02"),
        ("FOLLOW_DIG", "dig_03"),
    ]


def test_requested_cycle_count_wraps_the_fixed_dig_sequence() -> None:
    cycle = ResidentFixedCycle(_plan())
    directive = cycle.start(
        run_id="run-wrap",
        requested_cycles=2,
        first_dig_point_id="dig_03",
    )
    assert directive.target_id == "dig_03"

    for child, reason, steps in (
        ("follow", "SUCCEEDED", None),
        ("act", "STEP_BUDGET_REACHED", 130),
        ("follow", "SUCCEEDED", None),
        ("fixed_action", "SEQUENCE_COMPLETED", None),
    ):
        directive = cycle.record_child_result(
            child=child,
            outcome="SUCCEEDED",
            reason_code=reason,
            quiescence_confirmed=True,
            completed_steps=steps,
        )

    assert directive.stage == "FOLLOW_DIG"
    assert directive.target_id == "dig_01"


@pytest.mark.parametrize(
    ("child", "reason", "quiescent", "completed_steps", "expected_reason"),
    [
        ("follow", "TRACKING_TIMEOUT", True, None, "TRACKING_TIMEOUT"),
        ("follow", "SUCCEEDED", False, None, "CHILD_NOT_QUIESCENT"),
    ],
)
def test_child_failure_or_missing_quiescence_stops_without_next_directive(
    child: str,
    reason: str,
    quiescent: bool,
    completed_steps: int | None,
    expected_reason: str,
) -> None:
    cycle = ResidentFixedCycle(_plan())
    cycle.start(run_id="run-fail", requested_cycles=1)

    directive = cycle.record_child_result(
        child=child,
        outcome="FAILED" if reason != "SUCCEEDED" else "SUCCEEDED",
        reason_code=reason,
        quiescence_confirmed=quiescent,
        completed_steps=completed_steps,
    )

    assert directive is None
    assert cycle.snapshot.stage == "FAILED"
    assert cycle.snapshot.reason_code == expected_reason
    assert cycle.snapshot.terminal is True


def test_act_requires_the_exact_acknowledged_step_budget() -> None:
    cycle = ResidentFixedCycle(_plan())
    cycle.start(run_id="run-act", requested_cycles=1)
    cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )

    directive = cycle.record_child_result(
        child="act",
        outcome="SUCCEEDED",
        reason_code="STEP_BUDGET_REACHED",
        quiescence_confirmed=True,
        completed_steps=129,
    )

    assert directive is None
    assert cycle.snapshot.stage == "FAILED"
    assert cycle.snapshot.reason_code == "ACT_STEP_BUDGET_MISMATCH"


def test_coordinator_dispatches_the_whole_local_cycle_from_one_start() -> None:
    calls: list[tuple[object, ...]] = []

    class Driver:
        def start_follow(self, artifact):
            calls.append(("follow", artifact.trajectory_id))

        def activate_act(self, *, max_steps):
            calls.append(("act", max_steps))

        def start_fixed_action(self, behavior):
            calls.append(("fixed", behavior))

        def terminal_disarm(self):
            calls.append(("disarm",))

    coordinator = ResidentFixedCycleCoordinator(plan=_plan(), driver=Driver())

    coordinator.start(run_id="run-local", requested_cycles=1)
    coordinator.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )
    coordinator.record_child_result(
        child="act",
        outcome="SUCCEEDED",
        reason_code="STEP_BUDGET_REACHED",
        quiescence_confirmed=True,
        completed_steps=130,
    )
    coordinator.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )
    coordinator.record_child_result(
        child="fixed_action",
        outcome="SUCCEEDED",
        reason_code="SEQUENCE_COMPLETED",
        quiescence_confirmed=True,
    )

    assert calls == [
        ("follow", "field-dig_01-v1"),
        ("act", 130),
        ("follow", "field-dump-v1"),
        ("fixed", "ExecuteDump"),
        ("disarm",),
    ]
    assert coordinator.snapshot.stage == "COMPLETED"


def test_cancel_disarms_before_publishing_terminal_state() -> None:
    events: list[str] = []

    class Driver:
        def start_follow(self, _artifact):
            events.append("follow")

        def activate_act(self, *, max_steps):
            raise AssertionError(max_steps)

        def start_fixed_action(self, behavior):
            raise AssertionError(behavior)

        def terminal_disarm(self):
            events.append("disarm")

    coordinator = ResidentFixedCycleCoordinator(plan=_plan(), driver=Driver())
    coordinator.start(run_id="run-cancel", requested_cycles=1)

    coordinator.cancel()

    assert events == ["follow", "disarm"]
    assert coordinator.snapshot.stage == "CANCELLED"
    assert coordinator.snapshot.terminal is True


def test_local_dispatch_failure_fails_cycle_and_requests_terminal_disarm() -> None:
    events: list[str] = []

    class Driver:
        def start_follow(self, _artifact):
            events.append("follow")
            raise RuntimeError("driver detail must not escape")

        def activate_act(self, *, max_steps):
            raise AssertionError(max_steps)

        def start_fixed_action(self, behavior):
            raise AssertionError(behavior)

        def terminal_disarm(self):
            events.append("disarm")

    coordinator = ResidentFixedCycleCoordinator(plan=_plan(), driver=Driver())

    with pytest.raises(RuntimeError, match="local dispatch failed") as raised:
        coordinator.start(run_id="run-dispatch-fail", requested_cycles=1)

    assert "driver detail" not in str(raised.value)
    assert events == ["follow", "disarm"]
    assert coordinator.snapshot.stage == "FAILED"
    assert coordinator.snapshot.reason_code == "LOCAL_DISPATCH_FAILED"
    assert coordinator.snapshot.terminal is True
