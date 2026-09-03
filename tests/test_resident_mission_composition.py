from __future__ import annotations

from pathlib import Path

import pytest

from edge_runtime.resident_fixed_cycle import (
    FixedCyclePlan,
    ResidentFixedCycle,
    load_fixed_cycle_plan,
)


ROOT = Path(__file__).parents[1]


def _plan(phases: list[dict[str, object]]) -> FixedCyclePlan:
    act_bindings = {
        str(phase["behavior_id"]): "a" * 64
        for phase in phases
        if str(phase["behavior_id"]).startswith("act_")
    }
    mission = {
        "schema_version": "resident_mission_definition.v1",
        "mission_id": "test_composed_mission",
        "entry_behavior": phases[0],
        "cycle_behaviors": phases[1:],
        "act_policy_bindings": act_bindings,
    }
    from edge_runtime._resident_mission_definition import ResidentMissionDefinition

    definition = ResidentMissionDefinition.from_mapping(mission)
    return FixedCyclePlan.from_mapping(
        {
            "schema_version": "resident_fixed_cycle_plan.v5",
            "plan_id": "declarative-mission-candidate",
            "validation_status": "candidate",
            "dig_sequence": ["dig_01", "dig_02"],
            "default_dig_group": "all",
            "dig_groups": {"all": ["dig_01", "dig_02"]},
            "source_catalog_sha256": "a" * 64,
            "target_catalog": {
                "catalog_id": "declarative-targets",
                "path": "/opt/excavator/declarative-targets.json",
                "sha256": "b" * 64,
            },
            "mission": mission,
            "mission_sha256": definition.sha256,
        },
        allow_candidate=True,
    )


def _phase(
    stage_id: str,
    behavior_id: str,
    *,
    target_role: str | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
    return {
        "stage_id": stage_id,
        "behavior_id": behavior_id,
        "target_role": target_role,
        "max_steps": max_steps,
    }


def _succeed(cycle: ResidentFixedCycle, directive):
    if directive.child == "follow":
        return cycle.record_child_result(
            child="follow",
            outcome="SUCCEEDED",
            reason_code="SUCCEEDED",
            quiescence_confirmed=True,
        )
    if directive.child == "act":
        return cycle.record_child_result(
            child="act",
            outcome="SUCCEEDED",
            reason_code="STEP_BUDGET_REACHED",
            quiescence_confirmed=True,
            completed_steps=directive.max_steps,
        )
    return cycle.record_child_result(
        child="fixed_action",
        outcome="SUCCEEDED",
        reason_code="SEQUENCE_COMPLETED",
        quiescence_confirmed=True,
    )


def test_mission_behavior_order_is_defined_only_by_the_plan() -> None:
    phases = [
        _phase("TRACK_DIG", "onnx_rl_tracking", target_role="current_dig"),
        _phase("DIG", "act_dig_lift", max_steps=125),
        _phase("TRACK_DUMP", "onnx_rl_tracking", target_role="dump"),
        _phase("DUMP", "fixed_dump"),
        _phase("RETURN_DIG", "onnx_rl_tracking", target_role="return_dig"),
    ]
    plan = _plan(phases)
    assert plan.mission.behavior_ids == (
        "onnx_rl_tracking",
        "act_dig_lift",
        "fixed_dump",
    )
    assert plan.mission.requires_act_worker is True
    assert plan.mission.act_worker_behavior_id == "act_dig_lift"
    assert plan.mission.act_worker_model_sha256 == "a" * 64
    assert plan.mission.trajectory_controller_backend == "onnx_rl"
    assert plan.mission.to_mapping()["cycle_behaviors"] == phases[1:]
    cycle = ResidentFixedCycle(plan)

    directive = cycle.start(run_id="run-composed", requested_cycles=1)
    observed = []
    while directive is not None:
        observed.append((directive.stage, directive.behavior_id))
        directive = _succeed(cycle, directive)

    assert observed == [
        ("TRACK_DIG", "onnx_rl_tracking"),
        ("DIG", "act_dig_lift"),
        ("TRACK_DUMP", "onnx_rl_tracking"),
        ("DUMP", "fixed_dump"),
        ("RETURN_DIG", "onnx_rl_tracking"),
    ]
    assert cycle.snapshot.stage == "COMPLETED"
    assert cycle.snapshot.completed_cycles == 1


def test_changing_only_phases_builds_the_engineering_reference() -> None:
    phases = [
        _phase("TRACK_DIG", "onnx_rl_tracking", target_role="current_dig"),
        _phase(
            "DIG_TRANSPORT_DUMP",
            "act_dig_transport_dump",
            max_steps=260,
        ),
        _phase("RETURN_DIG", "onnx_rl_tracking", target_role="return_dig"),
    ]
    cycle = ResidentFixedCycle(_plan(phases))

    directive = cycle.start(run_id="run-reference", requested_cycles=1)
    observed = []
    while directive is not None:
        observed.append((directive.stage, directive.behavior_id))
        directive = _succeed(cycle, directive)

    assert observed == [
        ("TRACK_DIG", "onnx_rl_tracking"),
        ("DIG_TRANSPORT_DUMP", "act_dig_transport_dump"),
        ("RETURN_DIG", "onnx_rl_tracking"),
    ]


@pytest.mark.parametrize(
    ("relative_plan", "mission_id", "expected_behaviors"),
    [
        (
            "deploy/v3b/catalog/candidate/fixed_cycle.candidate.json",
            "fixed_target_hybrid",
            [
                "onnx_rl_tracking",
                "act_dig_lift",
                "onnx_rl_tracking",
                "fixed_dump",
                "onnx_rl_tracking",
            ],
        ),
        (
            "deploy/v3b/classical-tracking/catalog/candidate/"
            "fixed_cycle.candidate.json",
            "classical_tracking_hybrid",
            [
                "cartesian_p_tracking",
                "act_dig_lift",
                "cartesian_p_tracking",
                "fixed_dump",
                "cartesian_p_tracking",
            ],
        ),
        (
            "deploy/v3b/fixed-dig/catalog/candidate/fixed_cycle.candidate.json",
            "fixed_dig_hybrid",
            [
                "onnx_rl_tracking",
                "fixed_dig",
                "onnx_rl_tracking",
                "fixed_dump",
                "onnx_rl_tracking",
            ],
        ),
        (
            "deploy/v3b/act-dig-transport-dump-reference/catalog/candidate/"
            "fixed_cycle.candidate.json",
            "engineering_act_transport_reference",
            [
                "onnx_rl_tracking",
                "act_dig_transport_dump",
                "onnx_rl_tracking",
            ],
        ),
    ],
)
def test_checked_in_missions_are_only_behavior_compositions(
    relative_plan: str,
    mission_id: str,
    expected_behaviors: list[str],
) -> None:
    plan = load_fixed_cycle_plan(ROOT / relative_plan, allow_candidate=True)

    assert plan.mission.mission_id == mission_id
    assert [item.behavior_id for item in plan.mission.behaviors] == (
        expected_behaviors
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda phases: phases[1].update(behavior_id="unknown_policy"),
            "unsupported resident behavior_id",
        ),
        (
            lambda phases: phases[-1].update(target_role="dump"),
            "last cycle behavior must track return_dig",
        ),
        (
            lambda phases: phases[2].update(
                behavior_id="cartesian_p_tracking"
            ),
            "one Mission must use one trajectory controller backend",
        ),
        (
            lambda phases: phases[1].update(max_steps=None),
            "max_steps",
        ),
    ],
)
def test_invalid_mission_compositions_fail_before_runtime_resources(
    mutate,
    message: str,
) -> None:
    phases = [
        _phase("TRACK_DIG", "onnx_rl_tracking", target_role="current_dig"),
        _phase("DIG", "act_dig_lift", max_steps=125),
        _phase("TRACK_DUMP", "onnx_rl_tracking", target_role="dump"),
        _phase("DUMP", "fixed_dump"),
        _phase("RETURN_DIG", "onnx_rl_tracking", target_role="return_dig"),
    ]
    mutate(phases)

    with pytest.raises(ValueError, match=message):
        _plan(phases)
