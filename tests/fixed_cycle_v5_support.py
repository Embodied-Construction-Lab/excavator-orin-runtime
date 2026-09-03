from __future__ import annotations

import hashlib
import json
from pathlib import Path

from edge_runtime.resident_fixed_cycle import (
    FixedCyclePlan,
    load_fixed_cycle_registry,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MISSION_FILES = {
    "fixed_target_hybrid": "fixed_target_hybrid.json",
    "classical_tracking_hybrid": "classical_tracking_hybrid.json",
    "fixed_dig_hybrid": "fixed_dig_hybrid.json",
    "engineering_act_transport_reference": (
        "engineering_act_transport_reference.json"
    ),
}


def mission_document(mission_id: str) -> dict[str, object]:
    filename = MISSION_FILES[mission_id]
    return json.loads((REPOSITORY / "deploy/missions" / filename).read_text())


def plan_document(
    *,
    mission_id: str = "fixed_target_hybrid",
    point_ids: tuple[str, ...] = ("dig_01", "dig_02", "dig_03"),
    dig_groups: dict[str, list[str]] | None = None,
    catalog_path: Path = Path("/opt/excavator/target_catalog.json"),
    catalog_sha256: str = "a" * 64,
    validation_status: str = "field_validated",
) -> dict[str, object]:
    mission = mission_document(mission_id)
    mission_sha256 = _canonical_sha256(mission)
    groups = dig_groups or {"all": list(point_ids)}
    return {
        "schema_version": "resident_fixed_cycle_plan.v5",
        "plan_id": f"{mission_id}-test-plan",
        "validation_status": validation_status,
        "dig_sequence": list(point_ids),
        "default_dig_group": next(iter(groups)),
        "dig_groups": groups,
        "source_catalog_sha256": "b" * 64,
        "target_catalog": {
            "catalog_id": f"{mission_id}-test-catalog",
            "path": str(catalog_path),
            "sha256": catalog_sha256,
        },
        "mission": mission,
        "mission_sha256": mission_sha256,
    }


def write_deployment(
    tmp_path: Path,
    *,
    mission_id: str = "fixed_target_hybrid",
    point_ids: tuple[str, ...] = ("dig_01", "dig_02", "dig_03"),
    validation_status: str = "field_validated",
) -> tuple[FixedCyclePlan, object]:
    mission = mission_document(mission_id)
    mission_sha256 = _canonical_sha256(mission)
    catalog = {
        "schema_version": "resident_fixed_target_catalog.v1",
        "catalog_id": f"{mission_id}-test-catalog",
        "validation_status": validation_status,
        "frame_id": "machine_root_ros",
        "mission_id": mission_id,
        "mission_sha256": mission_sha256,
        "control_stage": "commissioning",
        "workspace_constraint": (
            "disabled_by_operator"
            if validation_status == "candidate"
            else "field_validated"
        ),
        "dig_points": {
            point_id: [1.0, 0.2 - index * 0.2, 0.0]
            for index, point_id in enumerate(point_ids)
        },
        "dump_target": [-0.2, -0.9, 0.1],
        "waypoint_tolerance_m": 0.25,
        "intermediate_waypoint_tolerance_m": 0.4,
        "waypoint_dwell_s": 0.0,
        "tracking_timeout_s": 60.0,
    }
    payload = json.dumps(
        catalog, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    catalog_path = tmp_path / "target_catalog.json"
    catalog_path.write_bytes(payload)
    document = plan_document(
        mission_id=mission_id,
        point_ids=point_ids,
        catalog_path=catalog_path,
        catalog_sha256=hashlib.sha256(payload).hexdigest(),
        validation_status=validation_status,
    )
    plan = FixedCyclePlan.from_mapping(
        document, allow_candidate=validation_status == "candidate"
    )
    return plan, load_fixed_cycle_registry(
        plan, allow_candidate=validation_status == "candidate"
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
