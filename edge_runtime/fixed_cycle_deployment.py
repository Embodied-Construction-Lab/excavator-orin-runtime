"""Build and promote immutable V3-A fixed-trajectory deployments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .resident_fixed_cycle import (
    load_fixed_cycle_plan,
    load_fixed_cycle_registry,
)


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
    demo_path: str | Path,
    output_dir: str | Path,
    deployed_root: str | Path,
    act_max_steps: int = 130,
    intermediate_waypoint_tolerance_m: float = 0.40,
) -> Path:
    """Create candidate endpoint trajectories from authoritative Airy configs."""

    mission_source = Path(mission_path)
    demo_source = Path(demo_path)
    mission = _load_object(mission_source)
    demo = _load_object(demo_source)
    if mission.get("schema_version") != "excavation_mission.v1":
        raise ValueError("unsupported excavation mission schema")
    if demo.get("schema_version") != "excavation_demo.v1":
        raise ValueError("unsupported excavation demo schema")
    if mission.get("frame_id") != "machine_root_ros" or demo.get("frame_id") != "machine_root_ros":
        raise ValueError("fixed cycle source frame must be machine_root_ros")
    if isinstance(act_max_steps, bool) or not isinstance(act_max_steps, int) or not 1 <= act_max_steps <= 2000:
        raise ValueError("act_max_steps must be within [1, 2000]")

    points = _dig_points(demo.get("dig_points"))
    dump = _position("dump target", demo.get("dump_target"))
    mission_dump = _position("mission dump target", _object(mission.get("targets"), "mission targets").get("dump"))
    if dump != mission_dump:
        raise ValueError("mission and demo dump targets do not match")
    limits = _limits(mission.get("limits"), demo.get("limits"))
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
    mission_bytes = mission_source.read_bytes()
    demo_bytes = demo_source.read_bytes()
    source_sha = hashlib.sha256(mission_bytes + b"\0" + demo_bytes).hexdigest()
    mission_id = _text("mission_id", mission.get("mission_id"))

    targets = {**points, "dump": dump}
    artifacts: dict[str, dict[str, Any]] = {}
    for target_id, position in targets.items():
        phase = "dump" if target_id == "dump" else "dig"
        filename = f"trajectory.{target_id}.candidate.json"
        trajectory_id = f"v3a-{target_id}-candidate"
        template = _trajectory_document(
            trajectory_id=trajectory_id,
            status="candidate",
            phase=phase,
            mission_id=mission_id,
            mission_sha256=source_sha,
            position=position,
            limits=limits,
        )
        payload = _canonical_bytes(template)
        (output / filename).write_bytes(payload)
        artifacts[target_id] = {
            "trajectory_id": trajectory_id,
            "phase": phase,
            "path": str(deployed / filename),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    plan = {
        "schema_version": "resident_fixed_cycle_plan.v1",
        "plan_id": "v3a-fixed-cycle-candidate",
        "validation_status": "candidate",
        "dig_sequence": list(points),
        "act_max_steps": act_max_steps,
        "trajectories": artifacts,
    }
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
    registry = load_fixed_cycle_registry(candidate, allow_candidate=True)
    record = _validation_record(validation_record_path, candidate)
    output = _new_directory(output_dir)
    deployed = Path(deployed_root)
    if not deployed.is_absolute():
        raise ValueError("deployed_root must be absolute")
    record_payload = _canonical_bytes(record)
    (output / "commissioning_record.json").write_bytes(record_payload)
    record_sha = hashlib.sha256(record_payload).hexdigest()

    artifacts: dict[str, dict[str, Any]] = {}
    for target_id in (*candidate.dig_sequence, "dump"):
        item = registry[target_id]
        filename = f"trajectory.{target_id}.field.json"
        document = _trajectory_document(
            trajectory_id=f"v3a-{target_id}-field",
            status="field_validated",
            phase=item.phase,
            mission_id=item.mission_id,
            mission_sha256=item.mission_sha256,
            position=item.waypoints[-1],
            limits={
                "waypoint_tolerance_m": item.waypoint_tolerance_m,
                "intermediate_waypoint_tolerance_m": (
                    item.intermediate_waypoint_tolerance_m
                ),
                "waypoint_dwell_s": item.waypoint_dwell_s,
                "tracking_timeout_s": item.tracking_timeout_s,
            },
        )
        payload = _canonical_bytes(document)
        (output / filename).write_bytes(payload)
        artifacts[target_id] = {
            "trajectory_id": document["trajectory_id"],
            "phase": item.phase,
            "path": str(deployed / filename),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    plan = {
        "schema_version": "resident_fixed_cycle_plan.v1",
        "plan_id": f"v3a-fixed-cycle-field-{record_sha}",
        "validation_status": "field_validated",
        "dig_sequence": list(candidate.dig_sequence),
        "act_max_steps": candidate.act_max_steps,
        "trajectories": artifacts,
    }
    plan_path = output / "fixed_cycle.field.json"
    plan_path.write_bytes(_canonical_bytes(plan))
    return plan_path


def _trajectory_document(
    *,
    trajectory_id: str,
    status: str,
    phase: str,
    mission_id: str,
    mission_sha256: str,
    position: tuple[float, float, float],
    limits: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "resident_fixed_trajectory.v1",
        "trajectory_id": trajectory_id,
        "validation_status": status,
        "phase": phase,
        "frame_id": "machine_root_ros",
        "mission_id": mission_id,
        "mission_sha256": mission_sha256,
        "task_mode": "CarryMaterial" if phase == "dump" else "MoveToDig",
        "control_stage": "commissioning",
        "workspace_constraint": "disabled_by_operator" if status == "candidate" else "field_validated",
        "waypoints": [list(position)],
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
    expected = [*candidate.dig_sequence, "dump"]
    if value.get("validated_targets") != expected:
        raise ValueError("validation record must cover every fixed target")
    return value


def _dig_points(value: Any) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("dig_points must be a non-empty list")
    points: dict[str, tuple[float, float, float]] = {}
    for item in value:
        mapping = _object(item, "dig point")
        point_id = _text("point_id", mapping.get("point_id"))
        if point_id in points:
            raise ValueError("dig point ids must be unique")
        points[point_id] = _position(point_id, mapping)
    return points


def _limits(mission: Any, demo: Any) -> dict[str, float]:
    mission_limits = _object(mission, "mission limits")
    demo_limits = _object(demo, "demo limits")
    result = {
        "waypoint_tolerance_m": _positive("waypoint_tolerance_m", mission_limits.get("waypoint_tolerance_m")),
        "waypoint_dwell_s": _nonnegative("waypoint_dwell_s", mission_limits.get("waypoint_dwell_s")),
        "tracking_timeout_s": _positive("tracking_timeout_s", mission_limits.get("tracking_timeout_s")),
    }
    if any(demo_limits.get(key) != value for key, value in result.items()):
        raise ValueError("mission and demo trajectory limits do not match")
    return result


def _position(name: str, value: Any) -> tuple[float, float, float]:
    mapping = _object(value, name)
    raw = mapping.get("position_m")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{name} position_m must contain three values")
    result = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} position_m must be finite")
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
