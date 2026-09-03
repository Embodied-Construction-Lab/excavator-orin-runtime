"""Build and promote immutable catalog-driven fixed-cycle deployments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ._resident_mission_definition import ResidentMissionDefinition
from .resident_fixed_cycle import load_fixed_cycle_plan, load_fixed_cycle_registry


PROMOTION_AUTHORIZATION = "PROMOTE_V3A_FIELD_VALIDATED_TRAJECTORIES"
_VALIDATION_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "operator_id",
        "tested_at",
        "result",
        "validated_targets",
    }
)


def build_candidate_deployment(
    *,
    mission_path: str | Path,
    mission_definition_path: str | Path,
    dig_point_catalog_path: str | Path,
    output_dir: str | Path,
    deployed_root: str | Path,
    intermediate_waypoint_tolerance_m: float = 0.40,
) -> Path:
    """Create one candidate target catalog from authoritative Airy configs."""

    mission_source = Path(mission_path)
    mission = _load_object(mission_source)
    definition_source = Path(mission_definition_path)
    definition = ResidentMissionDefinition.from_mapping(
        _load_object(definition_source)
    )
    if mission.get("schema_version") != "excavation_mission.v1":
        raise ValueError("unsupported excavation mission schema")
    if mission.get("frame_id") != "machine_root_ros":
        raise ValueError("fixed cycle source frame must be machine_root_ros")

    catalog_source = Path(dig_point_catalog_path)
    catalog_source_bytes = catalog_source.read_bytes()
    points, default_dig_group, dig_groups = _dig_point_catalog(
        _load_object(catalog_source)
    )
    dump = _position(
        "mission dump target",
        _object(mission.get("targets"), "mission targets").get("dump"),
    )
    limits = _limits(mission.get("limits"))
    intermediate_tolerance = _positive(
        "intermediate_waypoint_tolerance_m",
        intermediate_waypoint_tolerance_m,
    )
    if intermediate_tolerance < limits["waypoint_tolerance_m"]:
        raise ValueError(
            "intermediate_waypoint_tolerance_m must be no smaller than "
            "waypoint_tolerance_m"
        )
    limits["intermediate_waypoint_tolerance_m"] = intermediate_tolerance
    output = _new_directory(output_dir)
    deployed = Path(deployed_root)
    if not deployed.is_absolute():
        raise ValueError("deployed_root must be absolute")
    deployment_prefix = definition.mission_id

    catalog_filename = "target_catalog.candidate.json"
    catalog_id = f"{deployment_prefix}-targets-candidate"
    target_catalog = _target_catalog_document(
        catalog_id=catalog_id,
        status="candidate",
        mission_id=definition.mission_id,
        mission_sha256=definition.sha256,
        points=points,
        dump=dump,
        limits=limits,
    )
    catalog_payload = _canonical_bytes(target_catalog)
    (output / catalog_filename).write_bytes(catalog_payload)
    plan = {
        "schema_version": "resident_fixed_cycle_plan.v5",
        "plan_id": f"{deployment_prefix}-fixed-cycle-candidate",
        "validation_status": "candidate",
        "dig_sequence": list(points),
        "source_catalog_sha256": hashlib.sha256(
            catalog_source_bytes
        ).hexdigest(),
        "target_catalog": {
            "catalog_id": catalog_id,
            "path": str(deployed / catalog_filename),
            "sha256": hashlib.sha256(catalog_payload).hexdigest(),
        },
        "mission_sha256": definition.sha256,
    }
    plan["mission"] = definition.to_mapping()
    plan["default_dig_group"] = default_dig_group
    plan["dig_groups"] = dig_groups
    plan_path = output / "fixed_cycle.candidate.json"
    plan_path.write_bytes(_canonical_bytes(plan))
    return plan_path


def promote_candidate_deployment(
    *,
    candidate_plan_path: str | Path,
    validation_record_path: str | Path,
    output_dir: str | Path,
    deployed_root: str | Path,
    authorization: str,
) -> Path:
    """Promote a verified candidate only after an explicit passed record."""

    if authorization != PROMOTION_AUTHORIZATION:
        raise ValueError("exact promotion authorization is required")
    candidate = load_fixed_cycle_plan(candidate_plan_path, allow_candidate=True)
    if candidate.validation_status != "candidate":
        raise ValueError("promotion input must be a candidate plan")
    load_fixed_cycle_registry(candidate, allow_candidate=True)
    record = _validation_record(validation_record_path, candidate)
    output = _new_directory(output_dir)
    deployed = Path(deployed_root)
    if not deployed.is_absolute():
        raise ValueError("deployed_root must be absolute")
    record_payload = _canonical_bytes(record)
    (output / "commissioning_record.json").write_bytes(record_payload)
    record_sha = hashlib.sha256(record_payload).hexdigest()

    deployment_prefix = candidate.mission.mission_id
    candidate_catalog = candidate.target_catalog
    if candidate_catalog is None:
        raise ValueError("promotion requires a catalog-driven candidate plan")
    source_catalog = _load_object(candidate_catalog.path)
    catalog_filename = "target_catalog.field.json"
    catalog_id = f"{deployment_prefix}-targets-field-{record_sha}"
    field_catalog = dict(source_catalog)
    field_catalog.update(
        {
            "catalog_id": catalog_id,
            "validation_status": "field_validated",
            "workspace_constraint": "field_validated",
        }
    )
    catalog_payload = _canonical_bytes(field_catalog)
    (output / catalog_filename).write_bytes(catalog_payload)
    plan = {
        "schema_version": "resident_fixed_cycle_plan.v5",
        "plan_id": f"{deployment_prefix}-fixed-cycle-field-{record_sha}",
        "validation_status": "field_validated",
        "dig_sequence": list(candidate.dig_sequence),
        "source_catalog_sha256": candidate.source_catalog_sha256,
        "mission": candidate.mission.to_mapping(),
        "mission_sha256": candidate.mission_sha256,
        "default_dig_group": candidate.default_dig_group,
        "dig_groups": {
            group_id: list(point_ids)
            for group_id, point_ids in candidate.dig_groups.items()
        },
        "target_catalog": {
            "catalog_id": catalog_id,
            "path": str(deployed / catalog_filename),
            "sha256": hashlib.sha256(catalog_payload).hexdigest(),
        },
    }
    plan_path = output / "fixed_cycle.field.json"
    plan_path.write_bytes(_canonical_bytes(plan))
    return plan_path


def _target_catalog_document(
    *,
    catalog_id: str,
    status: str,
    mission_id: str,
    mission_sha256: str,
    points: Mapping[str, tuple[float, float, float]],
    dump: tuple[float, float, float],
    limits: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "resident_fixed_target_catalog.v1",
        "catalog_id": catalog_id,
        "validation_status": status,
        "frame_id": "machine_root_ros",
        "mission_id": mission_id,
        "mission_sha256": mission_sha256,
        "control_stage": "commissioning",
        "workspace_constraint": "disabled_by_operator" if status == "candidate" else "field_validated",
        "dig_points": {
            point_id: list(position) for point_id, position in points.items()
        },
        "dump_target": list(dump),
        "waypoint_tolerance_m": limits["waypoint_tolerance_m"],
        "intermediate_waypoint_tolerance_m": limits[
            "intermediate_waypoint_tolerance_m"
        ],
        "waypoint_dwell_s": limits["waypoint_dwell_s"],
        "tracking_timeout_s": limits["tracking_timeout_s"],
    }


def _validation_record(path: str | Path, candidate: Any) -> dict[str, Any]:
    value = _load_object(Path(path))
    if set(value) != _VALIDATION_FIELDS or value.get("schema_version") != "v3a_fixed_cycle_validation.v1":
        raise ValueError("fixed cycle validation record fields are invalid")
    for field in ("run_id", "operator_id", "tested_at"):
        _text(field, value.get(field))
    if value.get("result") != "passed":
        raise ValueError("fixed cycle validation result must be passed")
    expected = list(candidate.trajectory_target_ids)
    if value.get("validated_targets") != expected:
        raise ValueError("validation record must cover every fixed target")
    return value


def _dig_point_catalog(
    value: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, float, float]], str, dict[str, list[str]]]:
    expected_fields = {
        "schema_version",
        "frame_id",
        "dig_points",
        "default_dig_group",
        "dig_groups",
    }
    if set(value) != expected_fields:
        raise ValueError("dig point catalog fields are invalid")
    if value["schema_version"] != "excavation_dig_point_catalog.v1":
        raise ValueError("unsupported dig point catalog schema")
    if value["frame_id"] != "machine_root_ros":
        raise ValueError("dig point catalog frame must be machine_root_ros")
    raw_points = value["dig_points"]
    if not isinstance(raw_points, Mapping) or not raw_points:
        raise ValueError("dig_points must be a non-empty object")
    points = {
        _text("dig point id", point_id): _position_values(
            f"dig point {point_id}", position
        )
        for point_id, position in raw_points.items()
    }
    raw_groups = value["dig_groups"]
    if not isinstance(raw_groups, Mapping) or not raw_groups:
        raise ValueError("dig_groups must be a non-empty object")
    groups: dict[str, list[str]] = {}
    for raw_group_id, raw_point_ids in raw_groups.items():
        group_id = _text("dig group id", raw_group_id)
        if not isinstance(raw_point_ids, list) or not raw_point_ids:
            raise ValueError("each dig group must be a non-empty list")
        point_ids = [_text("dig group point id", item) for item in raw_point_ids]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("dig group point ids must be unique")
        if not set(point_ids) <= set(points):
            raise ValueError("dig group references an unknown point")
        groups[group_id] = point_ids
    if groups.get("all") != list(points):
        raise ValueError("dig_groups.all must exactly match dig_points order")
    default_group = _text("default_dig_group", value["default_dig_group"])
    if default_group not in groups:
        raise ValueError("default_dig_group is not defined")
    return points, default_group, groups


def _limits(mission: Any) -> dict[str, float]:
    mission_limits = _object(mission, "mission limits")
    return {
        "waypoint_tolerance_m": _positive("waypoint_tolerance_m", mission_limits.get("waypoint_tolerance_m")),
        "waypoint_dwell_s": _nonnegative("waypoint_dwell_s", mission_limits.get("waypoint_dwell_s")),
        "tracking_timeout_s": _positive("tracking_timeout_s", mission_limits.get("tracking_timeout_s")),
    }


def _position(name: str, value: Any) -> tuple[float, float, float]:
    mapping = _object(value, name)
    raw = mapping.get("position_m")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{name} position_m must contain three values")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} position_m must be finite")
    return result  # type: ignore[return-value]


def _position_values(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_object(value, str(path)))


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _positive(name: str, value: Any) -> float:
    number = _nonnegative(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _new_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    return output


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
