import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from edge_runtime.fixed_cycle_deployment import (
    PROMOTION_AUTHORIZATION,
    build_candidate_deployment,
    promote_candidate_deployment,
)
from edge_runtime.resident_fixed_cycle import (
    load_fixed_cycle_plan,
    load_fixed_cycle_registry,
)


def _write_sources(root: Path) -> tuple[Path, Path]:
    mission = {
        "schema_version": "excavation_mission.v1",
        "mission_id": "field_cycle_001",
        "frame_id": "machine_root_ros",
        "targets": {"dump": {"position_m": [-0.2, -0.9, 0.1]}},
        "limits": {
            "waypoint_tolerance_m": 0.25,
            "waypoint_dwell_s": 0.0,
            "tracking_timeout_s": 60.0,
        },
    }
    demo = {
        "schema_version": "excavation_demo.v1",
        "demo_id": "field_demo_001",
        "frame_id": "machine_root_ros",
        "dig_points": [
            {"point_id": "dig_01", "position_m": [1.0, 0.26, 0.0]},
            {"point_id": "dig_02", "position_m": [1.0, 0.0, 0.0]},
            {"point_id": "dig_03", "position_m": [1.0, -0.26, 0.0]},
        ],
        "dump_target": {"position_m": [-0.2, -0.9, 0.1]},
        "limits": mission["limits"],
    }
    mission_path = root / "excavation_cycle.json"
    demo_path = root / "excavation_demo.json"
    mission_path.write_text(json.dumps(mission), encoding="utf-8")
    demo_path.write_text(json.dumps(demo), encoding="utf-8")
    return mission_path, demo_path


def _write_eight_point_catalog(root: Path) -> Path:
    points = {
        "dig_near_01": [1.0, 0.40, 0.0],
        "dig_near_02": [1.0, 0.15, 0.0],
        "dig_near_03": [1.0, -0.10, 0.0],
        "dig_near_04": [1.0, -0.35, 0.0],
        "dig_far_01": [1.3, 0.40, 0.0],
        "dig_far_02": [1.3, 0.15, 0.0],
        "dig_far_03": [1.3, -0.10, 0.0],
        "dig_far_04": [1.3, -0.35, 0.0],
    }
    path = root / "dig_point_catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_dig_point_catalog.v1",
                "frame_id": "machine_root_ros",
                "dig_points": points,
                "default_dig_group": "all",
                "dig_groups": {
                    "all": list(points),
                    "near": list(points)[:4],
                    "far": list(points)[4:],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_mission_definition(
    root: Path,
    mission_id: str = "fixed_target_hybrid",
) -> Path:
    tracking = "onnx_rl_tracking"
    if mission_id == "engineering_act_transport_reference":
        cycle = [
            {
                "stage_id": "DIG_TRANSPORT_DUMP",
                "behavior_id": "act_dig_transport_dump",
                "target_role": None,
                "max_steps": 260,
            },
            {
                "stage_id": "RETURN_DIG",
                "behavior_id": tracking,
                "target_role": "return_dig",
                "max_steps": None,
            },
        ]
    elif mission_id == "fixed_dig_hybrid":
        cycle = [
            {
                "stage_id": "FIXED_DIG",
                "behavior_id": "fixed_dig",
                "target_role": None,
                "max_steps": None,
            },
            {
                "stage_id": "TRACK_DUMP",
                "behavior_id": tracking,
                "target_role": "dump",
                "max_steps": None,
            },
            {
                "stage_id": "FIXED_DUMP",
                "behavior_id": "fixed_dump",
                "target_role": None,
                "max_steps": None,
            },
            {
                "stage_id": "RETURN_DIG",
                "behavior_id": tracking,
                "target_role": "return_dig",
                "max_steps": None,
            },
        ]
    else:
        cycle = [
            {
                "stage_id": "ACT_DIG",
                "behavior_id": "act_dig_lift",
                "target_role": None,
                "max_steps": 130,
            },
            {
                "stage_id": "TRACK_DUMP",
                "behavior_id": tracking,
                "target_role": "dump",
                "max_steps": None,
            },
            {
                "stage_id": "FIXED_DUMP",
                "behavior_id": "fixed_dump",
                "target_role": None,
                "max_steps": None,
            },
            {
                "stage_id": "RETURN_DIG",
                "behavior_id": tracking,
                "target_role": "return_dig",
                "max_steps": None,
            },
        ]
    act_policy_bindings = {}
    if mission_id == "engineering_act_transport_reference":
        act_policy_bindings["act_dig_transport_dump"] = "b" * 64
    elif mission_id != "fixed_dig_hybrid":
        act_policy_bindings["act_dig_lift"] = "a" * 64
    path = root / f"{mission_id}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "resident_mission_definition.v1",
                "mission_id": mission_id,
                "entry_behavior": {
                    "stage_id": "TRACK_DIG",
                    "behavior_id": tracking,
                    "target_role": "current_dig",
                    "max_steps": None,
                },
                "cycle_behaviors": cycle,
                "act_policy_bindings": act_policy_bindings,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_candidate_uses_one_catalog_for_points_and_groups(tmp_path: Path) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(tmp_path)
    catalog_path = _write_eight_point_catalog(tmp_path)
    output = tmp_path / "candidate"

    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=output,
        deployed_root=output,
    )

    plan = load_fixed_cycle_plan(plan_path, allow_candidate=True)
    registry = load_fixed_cycle_registry(plan, allow_candidate=True)
    plan_document = json.loads(plan_path.read_text(encoding="utf-8"))
    catalog_document = json.loads(
        (output / "target_catalog.candidate.json").read_text(encoding="utf-8")
    )
    assert plan_document["mission_sha256"] == plan.mission.sha256
    assert catalog_document["mission_id"] == plan.mission.mission_id
    assert catalog_document["mission_sha256"] == plan.mission.sha256
    assert plan.default_dig_group == "all"
    assert plan.dig_groups["near"] == tuple(list(plan.dig_sequence)[:4])
    assert plan.dig_groups["far"] == tuple(list(plan.dig_sequence)[4:])
    assert registry["dig_far_04"].waypoints == ((1.3, -0.35, 0.0),)


def test_plan_rejects_mission_content_drift_under_same_mission_id(
    tmp_path: Path,
) -> None:
    mission_path, _demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(tmp_path)
    output = tmp_path / "candidate"
    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=_write_eight_point_catalog(tmp_path),
        output_dir=output,
        deployed_root=output,
    )
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    document["mission"]["cycle_behaviors"][0]["max_steps"] = 2000
    plan_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="Mission sha256"):
        load_fixed_cycle_plan(plan_path, allow_candidate=True)


@pytest.mark.parametrize("point_count", [4, 5, 8])
def test_candidate_deployment_uses_one_catalog_snapshot_for_any_point_count(
    tmp_path: Path,
    point_count: int,
) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(tmp_path)
    point_ids = [f"dig_dynamic_{index:02d}" for index in range(1, point_count + 1)]
    catalog_path = tmp_path / "dynamic-catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_dig_point_catalog.v1",
                "frame_id": "machine_root_ros",
                "dig_points": {
                    point_id: [1.0 + index * 0.05, 0.4 - index * 0.1, 0.0]
                    for index, point_id in enumerate(point_ids)
                },
                "default_dig_group": "all",
                "dig_groups": {"all": point_ids},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "candidate"

    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=output,
        deployed_root=output,
    )

    plan = load_fixed_cycle_plan(plan_path, allow_candidate=True)
    registry = load_fixed_cycle_registry(plan, allow_candidate=True)
    assert plan.dig_sequence == tuple(point_ids)
    assert tuple(registry) == (*tuple(point_ids), "dump")
    assert registry[point_ids[-1]].waypoints[-1] == pytest.approx(
        (1.0 + (point_count - 1) * 0.05, 0.4 - (point_count - 1) * 0.1, 0.0)
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "fixed_cycle.candidate.json",
        "target_catalog.candidate.json",
    ]


def test_build_candidate_uses_fixed_endpoints_and_stays_uncommissioned(
    tmp_path: Path,
) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(tmp_path)
    catalog_path = _write_eight_point_catalog(tmp_path)
    output = tmp_path / "candidate"

    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=output,
        deployed_root=output,
    )

    with pytest.raises(ValueError, match="field_validated"):
        load_fixed_cycle_plan(plan_path)
    plan = load_fixed_cycle_plan(plan_path, allow_candidate=True)
    registry = load_fixed_cycle_registry(plan, allow_candidate=True)
    assert plan.validation_status == "candidate"
    assert len(plan.dig_sequence) == 8
    assert registry["dig_near_01"].waypoints == ((1.0, 0.4, 0.0),)
    assert registry["dig_far_04"].waypoints == ((1.3, -0.35, 0.0),)
    assert registry["dump"].waypoints == ((-0.2, -0.9, 0.1),)
    assert registry["dig_near_01"].waypoint_tolerance_m == 0.25
    assert registry["dig_near_01"].intermediate_waypoint_tolerance_m == 0.40


def test_build_act_dig_transport_dump_reference_candidate_omits_dump_trajectory(tmp_path: Path) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(
        tmp_path, "engineering_act_transport_reference"
    )
    catalog_path = _write_eight_point_catalog(tmp_path)
    output = tmp_path / "candidate"

    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=output,
        deployed_root=output,
    )

    plan = load_fixed_cycle_plan(plan_path, allow_candidate=True)
    registry = load_fixed_cycle_registry(plan, allow_candidate=True)
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "resident_fixed_cycle_plan.v5"
    assert document["source_catalog_sha256"] == hashlib.sha256(
        catalog_path.read_bytes()
    ).hexdigest()
    assert plan.mission.mission_id == "engineering_act_transport_reference"
    assert next(
        behavior.max_steps
        for behavior in plan.mission.behaviors
        if behavior.behavior_id == "act_dig_transport_dump"
    ) == 260
    assert tuple(registry) == tuple(plan.dig_sequence)
    assert len(registry) == 8
    assert not (output / "trajectory.dump.candidate.json").exists()


def test_build_fixed_dig_candidate_keeps_dump_target_and_profile(tmp_path: Path) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(tmp_path, "fixed_dig_hybrid")
    catalog_path = _write_eight_point_catalog(tmp_path)
    output = tmp_path / "candidate"

    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=output,
        deployed_root=output,
    )

    plan = load_fixed_cycle_plan(plan_path, allow_candidate=True)
    registry = load_fixed_cycle_registry(plan, allow_candidate=True)

    assert plan.mission.mission_id == "fixed_dig_hybrid"
    assert plan.trajectory_target_ids[-1] == "dump"
    assert "dump" in registry
    assert plan.plan_id.startswith("fixed_dig_hybrid-fixed-cycle-candidate")
    assert json.loads(plan_path.read_text(encoding="utf-8"))["target_catalog"][
        "catalog_id"
    ].startswith("fixed_dig_hybrid-targets-candidate")


def test_promote_act_dig_transport_dump_reference_requires_only_dig_targets(tmp_path: Path) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(
        tmp_path, "engineering_act_transport_reference"
    )
    catalog_path = _write_eight_point_catalog(tmp_path)
    candidate = tmp_path / "candidate"
    candidate_plan = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=candidate,
        deployed_root=candidate,
    )
    record = tmp_path / "validation.json"
    record.write_text(
        json.dumps(
            {
                "schema_version": "v3a_fixed_cycle_validation.v1",
                "run_id": "engine-off-act-reference-001",
                "operator_id": "zhaoshuai",
                "tested_at": "2026-08-28T10:00:00+08:00",
                "result": "passed",
                "validated_targets": list(
                    json.loads(candidate_plan.read_text(encoding="utf-8"))[
                        "dig_sequence"
                    ]
                ),
            }
        ),
        encoding="utf-8",
    )

    plan_path = promote_candidate_deployment(
        candidate_plan_path=candidate_plan,
        validation_record_path=record,
        output_dir=tmp_path / "field",
        deployed_root=tmp_path / "field",
        authorization=PROMOTION_AUTHORIZATION,
    )

    promoted = load_fixed_cycle_plan(plan_path)
    assert promoted.mission.mission_id == "engineering_act_transport_reference"
    assert not (tmp_path / "field" / "trajectory.dump.field.json").exists()


def test_promotion_requires_passed_record_and_rewrites_all_hashes(
    tmp_path: Path,
) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    definition_path = _write_mission_definition(tmp_path)
    catalog_path = _write_eight_point_catalog(tmp_path)
    candidate = tmp_path / "candidate"
    candidate_plan = build_candidate_deployment(
        mission_path=mission_path,
        mission_definition_path=definition_path,
        dig_point_catalog_path=catalog_path,
        output_dir=candidate,
        deployed_root=candidate,
    )
    record = tmp_path / "validation.json"
    record.write_text(
        json.dumps(
            {
                "schema_version": "v3a_fixed_cycle_validation.v1",
                "run_id": "engine-off-001",
                "operator_id": "zhaoshuai",
                "tested_at": "2026-08-25T10:00:00+08:00",
                "result": "passed",
                "validated_targets": [
                    *json.loads(candidate_plan.read_text(encoding="utf-8"))[
                        "dig_sequence"
                    ],
                    "dump",
                ],
            }
        ),
        encoding="utf-8",
    )
    field = tmp_path / "field"

    with pytest.raises(ValueError, match="promotion authorization"):
        promote_candidate_deployment(
            candidate_plan_path=candidate_plan,
            validation_record_path=record,
            output_dir=field,
            deployed_root=field,
            authorization="wrong",
        )

    plan_path = promote_candidate_deployment(
        candidate_plan_path=candidate_plan,
        validation_record_path=record,
        output_dir=field,
        deployed_root=field,
        authorization=PROMOTION_AUTHORIZATION,
    )
    plan = load_fixed_cycle_plan(plan_path)
    registry = load_fixed_cycle_registry(plan)
    record_copy = field / "commissioning_record.json"

    assert plan.validation_status == "field_validated"
    assert all(item.validation_status == "field_validated" for item in registry.values())
    assert all(
        item.intermediate_waypoint_tolerance_m == 0.40
        for item in registry.values()
    )
    assert record_copy.is_file()
    assert hashlib.sha256(record_copy.read_bytes()).hexdigest() in plan.plan_id


@pytest.mark.parametrize(
    "script_name",
    ["build_v3a_fixed_cycle_candidate.py", "promote_v3a_fixed_cycle.py"],
)
def test_deployment_cli_is_runnable_outside_the_repository(
    tmp_path: Path,
    script_name: str,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / script_name

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_candidate_builder_cli_requires_a_mission_definition(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "build_v3a_fixed_cycle_candidate.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--mission-definition" in result.stdout
    assert "--mission-profile" not in result.stdout
