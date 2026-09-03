import pytest

from fixed_cycle_v5_support import plan_document
from edge_runtime.resident_fixed_cycle import (
    FixedCyclePlan,
    ResidentFixedCycle,
    ResidentFixedCycleCoordinator,
)


def _plan_document() -> dict[str, object]:
    return plan_document()


def _plan() -> FixedCyclePlan:
    return FixedCyclePlan.from_mapping(_plan_document())


def _act_dig_transport_dump_reference_plan() -> FixedCyclePlan:
    return FixedCyclePlan.from_mapping(
        plan_document(mission_id="engineering_act_transport_reference")
    )


def _fixed_dig_plan() -> FixedCyclePlan:
    return FixedCyclePlan.from_mapping(
        plan_document(mission_id="fixed_dig_hybrid")
    )


def _eight_point_plan() -> FixedCyclePlan:
    point_ids = [
        *(f"dig_near_{index:02d}" for index in range(1, 5)),
        *(f"dig_far_{index:02d}" for index in range(1, 5)),
    ]
    document = plan_document(
        point_ids=tuple(point_ids),
        dig_groups={
            "all": point_ids,
            "near": point_ids[:4],
            "far": point_ids[4:],
        },
    )
    return FixedCyclePlan.from_mapping(document)


def test_selected_dig_group_is_frozen_for_the_complete_cycle() -> None:
    cycle = ResidentFixedCycle(_eight_point_plan())

    directive = cycle.start(
        run_id="run-near-only",
        requested_cycles=5,
        dig_group_id="near",
    )
    observed = [directive.target_id]
    for _ in range(4):
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
        observed.append(directive.target_id)

    assert observed == [
        "dig_near_01",
        "dig_near_02",
        "dig_near_03",
        "dig_near_04",
        "dig_near_01",
    ]
    assert cycle.snapshot.dig_group_id == "near"


def test_act_transport_reference_is_parallel_to_regime_factorized_mission() -> None:
    cycle = ResidentFixedCycle(_act_dig_transport_dump_reference_plan())

    directive = cycle.start(
        run_id="run-act-dig-transport-dump-reference",
        requested_cycles=1,
        first_dig_point_id="dig_01",
    )
    assert directive.stage == "TRACK_DIG"
    assert directive.target_id == "dig_01"
    assert cycle.snapshot.mission_id == "engineering_act_transport_reference"

    directive = cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )
    assert directive.stage == "ACT_DIG_TRANSPORT_DUMP"
    assert directive.child == "act"
    assert directive.max_steps == 260

    directive = cycle.record_child_result(
        child="act",
        outcome="SUCCEEDED",
        reason_code="STEP_BUDGET_REACHED",
        quiescence_confirmed=True,
        completed_steps=260,
    )
    assert directive.stage == "RETURN_DIG"
    assert directive.target_id == "dig_01"

    terminal = cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )
    assert terminal is None
    assert cycle.snapshot.stage == "COMPLETED"
    assert cycle.snapshot.completed_cycles == 1

    primary = ResidentFixedCycle(_plan())
    primary.start(run_id="run-primary", requested_cycles=1)
    primary_act = primary.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )
    assert primary_act.stage == "ACT_DIG"


def test_fixed_dig_profile_uses_execute_dig_before_follow_dump() -> None:
    cycle = ResidentFixedCycle(_fixed_dig_plan())

    directive = cycle.start(
        run_id="run-fixed-dig",
        requested_cycles=1,
        first_dig_point_id="dig_01",
    )
    assert directive.stage == "TRACK_DIG"
    assert directive.target_id == "dig_01"
    assert cycle.snapshot.mission_id == "fixed_dig_hybrid"

    directive = cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )
    assert directive.stage == "FIXED_DIG"
    assert directive.child == "fixed_action"
    assert directive.behavior == "ExecuteDig"

    directive = cycle.record_child_result(
        child="fixed_action",
        outcome="SUCCEEDED",
        reason_code="SEQUENCE_COMPLETED",
        quiescence_confirmed=True,
    )
    assert directive.stage == "TRACK_DUMP"
    assert directive.child == "follow"
    assert directive.target_id == "dump"


def test_legacy_profile_plans_are_not_executable() -> None:
    for legacy_schema in (
        "resident_fixed_cycle_plan.v1",
        "resident_fixed_cycle_plan.v2",
        "resident_fixed_cycle_plan.v3",
        "resident_fixed_cycle_plan.v4",
    ):
        document = _plan_document()
        document["schema_version"] = legacy_schema
        with pytest.raises(ValueError, match="unsupported.*schema"):
            FixedCyclePlan.from_mapping(document)


def test_three_cycles_advance_locally_without_per_stage_external_commands() -> None:
    cycle = ResidentFixedCycle(_plan())

    directive = cycle.start(
        run_id="run-001",
        requested_cycles=3,
        first_dig_point_id="dig_01",
    )
    observed = [(directive.stage, directive.target_id)]

    for expected_dig in ("dig_01", "dig_02", "dig_03"):
        assert directive.stage in {"TRACK_DIG", "RETURN_DIG"}
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
        assert directive.stage == "TRACK_DUMP"
        assert directive.target_id == "dump"

        directive = cycle.record_child_result(
            child="follow",
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
        observed.append((directive.stage, directive.behavior))
        assert directive.stage == "FIXED_DUMP"
        assert directive.behavior == "ExecuteDump"

        directive = cycle.record_child_result(
            child="fixed_action",
            outcome="SUCCEEDED",
            reason_code="SEQUENCE_COMPLETED",
            quiescence_confirmed=True,
        )
        if directive is not None:
            observed.append((directive.stage, directive.target_id))

    assert directive is not None
    assert directive.stage == "RETURN_DIG"
    assert directive.target_id == "dig_03"
    directive = cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )

    assert directive is None
    assert cycle.snapshot.stage == "COMPLETED"
    assert cycle.snapshot.completed_cycles == 3
    assert cycle.snapshot.outcome == "SUCCEEDED"
    assert [item for item in observed if item[0] in {"TRACK_DIG", "RETURN_DIG"}] == [
        ("TRACK_DIG", "dig_01"),
        ("RETURN_DIG", "dig_02"),
        ("RETURN_DIG", "dig_03"),
        ("RETURN_DIG", "dig_03"),
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

    assert directive.stage == "RETURN_DIG"
    assert directive.target_id == "dig_01"


def test_final_cycle_returns_to_its_dig_point_before_completion() -> None:
    cycle = ResidentFixedCycle(_plan())
    directive = cycle.start(
        run_id="run-final-return",
        requested_cycles=1,
        first_dig_point_id="dig_02",
    )

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

    assert directive is not None
    assert directive.stage == "RETURN_DIG"
    assert directive.target_id == "dig_02"
    assert cycle.snapshot.completed_cycles == 1
    assert cycle.snapshot.terminal is False

    terminal = cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )

    assert terminal is None
    assert cycle.snapshot.stage == "COMPLETED"
    assert cycle.snapshot.terminal is True


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


def test_act_accepts_a_confirmed_deadzone_chunk_before_the_step_budget() -> None:
    cycle = ResidentFixedCycle(_plan())
    cycle.start(run_id="run-act-early", requested_cycles=1)
    cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )

    directive = cycle.record_child_result(
        child="act",
        outcome="SUCCEEDED",
        reason_code="DEADZONE_CHUNK_REACHED",
        quiescence_confirmed=True,
        completed_steps=121,
    )

    assert directive is not None
    assert directive.stage == "TRACK_DUMP"
    assert cycle.snapshot.stage == "TRACK_DUMP"


def test_act_transport_reference_rejects_deadzone_completion_before_budget() -> None:
    cycle = ResidentFixedCycle(_act_dig_transport_dump_reference_plan())
    cycle.start(run_id="run-long-act", requested_cycles=1)
    cycle.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )

    directive = cycle.record_child_result(
        child="act",
        outcome="SUCCEEDED",
        reason_code="DEADZONE_CHUNK_REACHED",
        quiescence_confirmed=True,
        completed_steps=101,
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
    coordinator.record_child_result(
        child="follow",
        outcome="SUCCEEDED",
        reason_code="SUCCEEDED",
        quiescence_confirmed=True,
    )

    assert calls == [
        ("follow", "fixed_target_hybrid-test-catalog:dig_01"),
        ("act", 130),
        ("follow", "fixed_target_hybrid-test-catalog:dump"),
        ("fixed", "ExecuteDump"),
        ("follow", "fixed_target_hybrid-test-catalog:dig_01"),
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
    assert coordinator.snapshot.active_behavior_id == ""
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
    assert coordinator.snapshot.active_behavior_id == ""
    assert coordinator.snapshot.reason_code == "LOCAL_DISPATCH_FAILED"
    assert coordinator.snapshot.terminal is True
