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


def test_build_candidate_uses_fixed_endpoints_and_stays_uncommissioned(
    tmp_path: Path,
) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    output = tmp_path / "candidate"

    plan_path = build_candidate_deployment(
        mission_path=mission_path,
        demo_path=demo_path,
        output_dir=output,
        deployed_root=output,
        act_max_steps=130,
    )

    with pytest.raises(ValueError, match="field_validated"):
        load_fixed_cycle_plan(plan_path)
    plan = load_fixed_cycle_plan(plan_path, allow_candidate=True)
    registry = load_fixed_cycle_registry(plan, allow_candidate=True)
    assert plan.validation_status == "candidate"
    assert plan.dig_sequence == ("dig_01", "dig_02", "dig_03")
    assert registry["dig_01"].waypoints == ((1.0, 0.26, 0.0),)
    assert registry["dig_03"].waypoints == ((1.0, -0.26, 0.0),)
    assert registry["dump"].waypoints == ((-0.2, -0.9, 0.1),)
    assert registry["dig_01"].waypoint_tolerance_m == 0.25
    assert registry["dig_01"].intermediate_waypoint_tolerance_m == 0.40


def test_promotion_requires_passed_record_and_rewrites_all_hashes(
    tmp_path: Path,
) -> None:
    mission_path, demo_path = _write_sources(tmp_path)
    candidate = tmp_path / "candidate"
    candidate_plan = build_candidate_deployment(
        mission_path=mission_path,
        demo_path=demo_path,
        output_dir=candidate,
        deployed_root=candidate,
        act_max_steps=130,
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
                "validated_targets": ["dig_01", "dig_02", "dig_03", "dump"],
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
