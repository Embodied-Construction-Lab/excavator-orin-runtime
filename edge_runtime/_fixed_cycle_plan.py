"""Strict fixed-cycle plans, catalog value objects, and immutable loaders."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ._resident_mission_definition import (
    ResidentMissionDefinition,
)


SCHEMA_VERSION = "resident_fixed_cycle_plan.v5"
CATALOG_SCHEMA_VERSION = "resident_fixed_target_catalog.v1"
MAX_REQUESTED_CYCLES = 9
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "plan_id",
        "validation_status",
        "dig_sequence",
        "default_dig_group",
        "dig_groups",
        "source_catalog_sha256",
        "target_catalog",
        "mission",
        "mission_sha256",
    }
)
_CATALOG_ARTIFACT_FIELDS = frozenset({"catalog_id", "path", "sha256"})
_CATALOG_FIELDS = frozenset(
    {
        "schema_version",
        "catalog_id",
        "validation_status",
        "frame_id",
        "mission_id",
        "mission_sha256",
        "control_stage",
        "workspace_constraint",
        "dig_points",
        "dump_target",
        "waypoint_tolerance_m",
        "intermediate_waypoint_tolerance_m",
        "waypoint_dwell_s",
        "tracking_timeout_s",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _dig_groups(
    value: Any,
    dig_sequence: tuple[str, ...],
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dig_groups must be a non-empty object")
    known = frozenset(dig_sequence)
    groups: dict[str, tuple[str, ...]] = {}
    for raw_group_id, raw_point_ids in value.items():
        group_id = _identifier("dig group id", raw_group_id)
        if not isinstance(raw_point_ids, list) or not raw_point_ids:
            raise ValueError("each dig group must be a non-empty list")
        point_ids = tuple(
            _identifier("dig group point id", item) for item in raw_point_ids
        )
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("dig group point ids must be unique")
        if not set(point_ids) <= known:
            raise ValueError("dig group references an unknown point")
        groups[group_id] = point_ids
    if set(groups.get("all", ())) != known or groups.get("all") != dig_sequence:
        raise ValueError("dig_groups.all must exactly match dig_sequence")
    return MappingProxyType(groups)


def _validation_status(
    value: Any,
    *,
    allow_candidate: bool,
    subject: str,
) -> str:
    allowed = {"field_validated"}
    if allow_candidate:
        allowed.add("candidate")
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{subject} must be field_validated")
    return value


@dataclass(frozen=True)
class FixedTrajectoryArtifact:
    """One catalog-backed trajectory identity materialized for Runtime use."""

    trajectory_id: str
    phase: str
    path: Path
    sha256: str

@dataclass(frozen=True)
class FixedTargetCatalogArtifact:
    """One immutable target catalog used to materialize every endpoint."""

    catalog_id: str
    path: Path
    sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "FixedTargetCatalogArtifact":
        if not isinstance(value, Mapping) or set(value) != _CATALOG_ARTIFACT_FIELDS:
            raise ValueError("fixed target catalog artifact fields are invalid")
        path = Path(_text("target catalog path", value["path"]))
        if not path.is_absolute():
            raise ValueError("fixed target catalog path must be absolute")
        digest = value["sha256"]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("target catalog sha256 must be lowercase hexadecimal")
        return cls(
            catalog_id=_identifier("catalog_id", value["catalog_id"]),
            path=path,
            sha256=digest,
        )


@dataclass(frozen=True)
class FixedTrajectoryTemplate:
    """Immutable Follow inputs whose dynamic execution state stays local."""

    trajectory_id: str
    validation_status: str
    phase: str
    frame_id: str
    mission_id: str
    mission_sha256: str
    task_mode: str
    control_stage: str
    workspace_constraint: str
    waypoints: tuple[tuple[float, float, float], ...]
    waypoint_tolerance_m: float
    intermediate_waypoint_tolerance_m: float
    waypoint_dwell_s: float
    tracking_timeout_s: float

@dataclass(frozen=True)
class FixedCyclePlan:
    """Strict fixed-target plan loaded before any motion resource is opened."""

    plan_id: str
    validation_status: str
    dig_sequence: tuple[str, ...]
    default_dig_group: str
    dig_groups: Mapping[str, tuple[str, ...]]
    source_catalog_sha256: str
    trajectories: Mapping[str, FixedTrajectoryArtifact]
    target_catalog: FixedTargetCatalogArtifact
    mission: ResidentMissionDefinition
    mission_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        allow_candidate: bool = False,
    ) -> "FixedCyclePlan":
        if not isinstance(value, Mapping):
            raise ValueError("resident fixed cycle plan fields are invalid")
        schema_version = value.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported resident fixed cycle plan schema")
        if set(value) != _PLAN_FIELDS:
            raise ValueError("resident fixed cycle plan fields are invalid")
        mission = ResidentMissionDefinition.from_mapping(value["mission"])
        mission_sha256 = value["mission_sha256"]
        if mission_sha256 != mission.sha256:
            raise ValueError("resident Mission sha256 does not match definition")
        validation_status = _validation_status(
            value["validation_status"],
            allow_candidate=allow_candidate,
            subject="resident fixed cycle plan",
        )
        raw_sequence = value["dig_sequence"]
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise ValueError("dig_sequence must be a non-empty list")
        sequence = tuple(_identifier("dig point id", item) for item in raw_sequence)
        if len(set(sequence)) != len(sequence):
            raise ValueError("dig_sequence must contain unique target ids")
        default_dig_group = _identifier(
            "default_dig_group", value["default_dig_group"]
        )
        dig_groups = _dig_groups(value["dig_groups"], sequence)
        if default_dig_group not in dig_groups:
            raise ValueError("default_dig_group is not defined")
        source_catalog_sha256 = value.get("source_catalog_sha256", "")
        if (
            not isinstance(source_catalog_sha256, str)
            or _SHA256.fullmatch(source_catalog_sha256) is None
        ):
            raise ValueError("source_catalog_sha256 must be lowercase hexadecimal")
        target_catalog = FixedTargetCatalogArtifact.from_mapping(
            value["target_catalog"]
        )
        artifacts = {
            target_id: FixedTrajectoryArtifact(
                trajectory_id=f"{target_catalog.catalog_id}:{target_id}",
                phase="dump" if target_id == "dump" else "dig",
                path=target_catalog.path,
                sha256=target_catalog.sha256,
            )
            for target_id in (
                (*sequence, "dump")
                if "dump" in mission.target_roles
                else sequence
            )
        }
        return cls(
            plan_id=_identifier("plan_id", value["plan_id"]),
            validation_status=validation_status,
            dig_sequence=sequence,
            default_dig_group=default_dig_group,
            dig_groups=dig_groups,
            source_catalog_sha256=source_catalog_sha256,
            trajectories=MappingProxyType(artifacts),
            target_catalog=target_catalog,
            mission=mission,
            mission_sha256=mission_sha256,
        )

    @property
    def trajectory_target_ids(self) -> tuple[str, ...]:
        if "dump" in self.mission.target_roles:
            return (*self.dig_sequence, "dump")
        return self.dig_sequence


def load_fixed_cycle_plan(
    path: str | Path,
    *,
    allow_candidate: bool = False,
) -> FixedCyclePlan:
    """Load one strict plan without opening policy, camera, or serial resources."""

    plan_path = Path(path)
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    return FixedCyclePlan.from_mapping(
        document,
        allow_candidate=allow_candidate,
    )


def verify_fixed_cycle_artifacts(
    plan: FixedCyclePlan,
) -> Mapping[str, bytes]:
    """Load and hash every fixed trajectory before motion resources are opened.

    Returning the verified bytes lets the eventual Runtime materialize its
    in-memory registry without reopening mutable paths after this boundary.
    """

    if not isinstance(plan, FixedCyclePlan):
        raise ValueError("plan must be a FixedCyclePlan")
    payload = _read_regular_file_no_follow(plan.target_catalog.path)
    if hashlib.sha256(payload).hexdigest() != plan.target_catalog.sha256:
        raise ValueError("fixed target catalog sha256 mismatch")
    return MappingProxyType(
        {target_id: payload for target_id in plan.trajectory_target_ids}
    )


def load_fixed_cycle_registry(
    plan: FixedCyclePlan,
    *,
    allow_candidate: bool = False,
) -> Mapping[str, FixedTrajectoryTemplate]:
    """Build an immutable registry from already verified artifact bytes."""

    if plan.validation_status == "candidate" and not allow_candidate:
        raise ValueError("resident fixed cycle plan must be field_validated")
    payloads = verify_fixed_cycle_artifacts(plan)
    return _load_catalog_registry(
        plan,
        payloads[plan.trajectory_target_ids[0]],
    )


def _load_catalog_registry(
    plan: FixedCyclePlan,
    payload: bytes,
) -> Mapping[str, FixedTrajectoryTemplate]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("fixed target catalog is not valid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != _CATALOG_FIELDS:
        raise ValueError("fixed target catalog fields are invalid")
    if value["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise ValueError("unsupported fixed target catalog schema")
    artifact = plan.target_catalog
    assert artifact is not None
    if _identifier("catalog_id", value["catalog_id"]) != artifact.catalog_id:
        raise ValueError("fixed target catalog id does not match plan")
    validation_status = _validation_status(
        value["validation_status"],
        allow_candidate=plan.validation_status == "candidate",
        subject="fixed target catalog",
    )
    if validation_status != plan.validation_status:
        raise ValueError("fixed target catalog status does not match plan")
    if value["frame_id"] != "machine_root_ros":
        raise ValueError("fixed target catalog frame must be machine_root_ros")
    control_stage = _text("control_stage", value["control_stage"])
    workspace_constraint = _text(
        "workspace_constraint", value["workspace_constraint"]
    )
    expected_constraint = (
        "disabled_by_operator"
        if validation_status == "candidate"
        else "field_validated"
    )
    if control_stage != "commissioning" or workspace_constraint != expected_constraint:
        raise ValueError("fixed target catalog commissioning scope is invalid")
    mission_sha256 = value["mission_sha256"]
    if not isinstance(mission_sha256, str) or _SHA256.fullmatch(mission_sha256) is None:
        raise ValueError("mission_sha256 must be lowercase hexadecimal")
    if value["mission_id"] != plan.mission.mission_id:
        raise ValueError("target catalog mission_id does not match plan")
    if mission_sha256 != plan.mission_sha256:
        raise ValueError("target catalog mission_sha256 does not match plan")
    points = _target_positions(value["dig_points"])
    if set(points) != set(plan.dig_sequence):
        raise ValueError("target catalog points do not match dig_sequence")
    dump_target = _fixed_point("dump_target", value["dump_target"])
    waypoint_tolerance_m = _positive_float(
        "waypoint_tolerance_m", value["waypoint_tolerance_m"]
    )
    intermediate_tolerance = _positive_float(
        "intermediate_waypoint_tolerance_m",
        value["intermediate_waypoint_tolerance_m"],
    )
    if intermediate_tolerance < waypoint_tolerance_m:
        raise ValueError("intermediate waypoint tolerance is too small")
    registry = {}
    for target_id in plan.trajectory_target_ids:
        phase = "dump" if target_id == "dump" else "dig"
        registry[target_id] = FixedTrajectoryTemplate(
            trajectory_id=f"{artifact.catalog_id}:{target_id}",
            validation_status=validation_status,
            phase=phase,
            frame_id="machine_root_ros",
            mission_id=_identifier("mission_id", value["mission_id"]),
            mission_sha256=mission_sha256,
            task_mode="CarryMaterial" if phase == "dump" else "MoveToDig",
            control_stage=control_stage,
            workspace_constraint=workspace_constraint,
            waypoints=((dump_target if phase == "dump" else points[target_id]),),
            waypoint_tolerance_m=waypoint_tolerance_m,
            intermediate_waypoint_tolerance_m=intermediate_tolerance,
            waypoint_dwell_s=_nonnegative_float(
                "waypoint_dwell_s", value["waypoint_dwell_s"]
            ),
            tracking_timeout_s=_positive_float(
                "tracking_timeout_s", value["tracking_timeout_s"]
            ),
        )
    return MappingProxyType(registry)


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


def _fixed_point(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(_finite_float(f"{name} coordinate", axis) for axis in value)


def _target_positions(value: Any) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dig_points must be a non-empty object")
    return {
        _identifier("dig point id", target_id): _fixed_point(
            f"dig point {target_id}", position
        )
        for target_id, position in value.items()
    }


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_float(name: str, value: Any) -> float:
    converted = _finite_float(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _nonnegative_float(name: str, value: Any) -> float:
    converted = _finite_float(name, value)
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


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
