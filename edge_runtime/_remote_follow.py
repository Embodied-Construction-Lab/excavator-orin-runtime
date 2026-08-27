"""Validated Trajectory Snapshots and reusable Follow Runtime construction.

This private Module owns the immutable remote Follow input contract and the
Trajectory Controller construction seam.  The public compatibility surface
continues to live in :mod:`edge_runtime.remote`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .follow import EdgeFollowRuntime
from .kinematics import UrdfBucketTipKinematics
from .remote_validation import (
    boolean as _boolean,
    finite as _finite,
    nonnegative as _nonnegative,
    positive as _positive,
    sha256 as _sha256,
    text as _text,
    waypoints as _waypoints,
)
from .trajectory import MissionFollowLimits
from .trajectory_controller import (
    OnnxRlTrajectoryControllerAdapter,
    TrajectoryControllerFactory,
    build_trajectory_controller_builder,
)


_SNAPSHOT_FIELDS = {
    "trajectory_id",
    "trajectory_sha256",
    "frame_id",
    "created_at_s",
    "mission_id",
    "mission_sha256",
    "mission_phase",
    "task_mode",
    "planning_scope",
    "control_stage",
    "workspace_constraint",
    "execution_eligible",
    "source_bucket_tip_stamp_s",
    "source_local_map_stamp_s",
    "inputs_frozen_at_s",
    "valid_until_s",
    "input_source",
    "map_source",
    "clock_mode",
    "waypoints",
    "waypoint_tolerance_m",
    "waypoint_dwell_s",
    "tracking_timeout_s",
}
_TWO_LEVEL_SNAPSHOT_FIELDS = _SNAPSHOT_FIELDS | {
    "intermediate_waypoint_tolerance_m"
}
_DIGEST_FIELDS = _SNAPSHOT_FIELDS - {"trajectory_id", "trajectory_sha256"}
_MAX_CLOCK_SKEW_S = 0.5


def _compatibility_dependency(name: str, default: Any) -> Any:
    """Honor the historical ``edge_runtime.remote`` monkeypatch seam.

    ``EdgeFollowRuntimeFactory`` was originally implemented in the public
    compatibility Module, and deployment tests and downstream integrators may
    replace its construction dependencies there.  Resolve those two factories
    from the already-loaded facade without making the private Module import the
    facade (which would create an import cycle).
    """

    facade = sys.modules.get(f"{__package__}.remote")
    return getattr(facade, name, default) if facade is not None else default


def _controller_builder_from_config(
    config: Any,
    builder_factory: Callable[..., Callable[[], Any]],
) -> Callable[[], Any]:
    """Build once while retaining both historical deployment patch seams."""

    if (
        config.trajectory_controller_backend != "onnx_rl"
        or builder_factory is not build_trajectory_controller_builder
        or config.onnx_path is None
    ):
        return builder_factory(
            config.trajectory_controller_backend,
            onnx_path=config.onnx_path,
        )
    policy_type = _compatibility_dependency("OnnxPolicy", None)
    if policy_type is None:
        return builder_factory("onnx_rl", onnx_path=config.onnx_path)
    policy = policy_type(Path(config.onnx_path))
    factory = TrajectoryControllerFactory(
        {"onnx_rl": lambda: OnnxRlTrajectoryControllerAdapter(policy)}
    )
    return lambda: factory.create("onnx_rl")


@dataclass(frozen=True)
class FollowTrajectorySnapshot:
    trajectory_id: str
    trajectory_sha256: str
    frame_id: str
    created_at_s: float
    mission_id: str
    mission_sha256: str
    mission_phase: str
    task_mode: str
    planning_scope: str
    control_stage: str
    workspace_constraint: str
    execution_eligible: bool
    source_bucket_tip_stamp_s: float
    source_local_map_stamp_s: float
    inputs_frozen_at_s: float
    valid_until_s: float
    input_source: str
    map_source: str
    clock_mode: str
    waypoints: tuple[tuple[float, float, float], ...]
    waypoint_tolerance_m: float
    waypoint_dwell_s: float
    tracking_timeout_s: float
    intermediate_waypoint_tolerance_m: float | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        now_s: float,
    ) -> "FollowTrajectorySnapshot":
        if not isinstance(value, Mapping) or set(value) not in {
            frozenset(_SNAPSHOT_FIELDS),
            frozenset(_TWO_LEVEL_SNAPSHOT_FIELDS),
        }:
            raise ValueError("trajectory snapshot fields are invalid")
        snapshot = cls(
            trajectory_id=_text("trajectory_id", value["trajectory_id"]),
            trajectory_sha256=_sha256(
                "trajectory_sha256", value["trajectory_sha256"]
            ),
            frame_id=_text("frame_id", value["frame_id"]),
            created_at_s=_finite("created_at_s", value["created_at_s"]),
            mission_id=_text("mission_id", value["mission_id"]),
            mission_sha256=_sha256("mission_sha256", value["mission_sha256"]),
            mission_phase=_text("mission_phase", value["mission_phase"]),
            task_mode=_text("task_mode", value["task_mode"]),
            planning_scope=_text("planning_scope", value["planning_scope"]),
            control_stage=_text("control_stage", value["control_stage"]),
            workspace_constraint=_text(
                "workspace_constraint", value["workspace_constraint"]
            ),
            execution_eligible=_boolean(
                "execution_eligible", value["execution_eligible"]
            ),
            source_bucket_tip_stamp_s=_finite(
                "source_bucket_tip_stamp_s",
                value["source_bucket_tip_stamp_s"],
            ),
            source_local_map_stamp_s=_finite(
                "source_local_map_stamp_s",
                value["source_local_map_stamp_s"],
            ),
            inputs_frozen_at_s=_finite(
                "inputs_frozen_at_s", value["inputs_frozen_at_s"]
            ),
            valid_until_s=_finite("valid_until_s", value["valid_until_s"]),
            input_source=_text("input_source", value["input_source"]),
            map_source=_text("map_source", value["map_source"]),
            clock_mode=_text("clock_mode", value["clock_mode"]),
            waypoints=_waypoints(value["waypoints"]),
            waypoint_tolerance_m=_positive(
                "waypoint_tolerance_m", value["waypoint_tolerance_m"]
            ),
            waypoint_dwell_s=_nonnegative(
                "waypoint_dwell_s", value["waypoint_dwell_s"]
            ),
            tracking_timeout_s=_positive(
                "tracking_timeout_s", value["tracking_timeout_s"]
            ),
            intermediate_waypoint_tolerance_m=(
                _positive(
                    "intermediate_waypoint_tolerance_m",
                    value["intermediate_waypoint_tolerance_m"],
                )
                if "intermediate_waypoint_tolerance_m" in value
                else None
            ),
        )
        snapshot._validate_for_execution(now_s=now_s)
        if snapshot.computed_sha256() != snapshot.trajectory_sha256:
            raise ValueError(
                "trajectory_sha256 does not match Trajectory Snapshot content"
            )
        return snapshot

    def computed_sha256(self) -> str:
        digest_fields = set(_DIGEST_FIELDS)
        if self.intermediate_waypoint_tolerance_m is not None:
            digest_fields.add("intermediate_waypoint_tolerance_m")
        payload = {
            name: (
                [list(point) for point in self.waypoints]
                if name == "waypoints"
                else getattr(self, name)
            )
            for name in digest_fields
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_for_execution(self, *, now_s: float) -> None:
        current = _finite("now_s", now_s)
        if self.frame_id != "machine_root_ros":
            raise ValueError("frame_id must be machine_root_ros")
        expected_mode = {"dig": "MoveToDig", "dump": "CarryMaterial"}.get(
            self.mission_phase
        )
        if self.task_mode != expected_mode:
            raise ValueError("mission_phase and task_mode mismatch")
        if self.planning_scope != "execution_strict":
            raise ValueError("planning_scope must be execution_strict")
        if not self.execution_eligible:
            raise ValueError("execution_eligible must be true")
        if self.input_source != "live" or self.map_source != "live_local_map":
            raise ValueError("input_source/map_source must identify live inputs")
        if self.clock_mode != "ros_clock":
            raise ValueError("clock_mode must be ros_clock")
        if self.control_stage == "production":
            valid_workspace = self.workspace_constraint == "field_validated"
        elif self.control_stage == "commissioning":
            valid_workspace = self.workspace_constraint in {
                "disabled_by_operator",
                "field_validated",
            }
        else:
            raise ValueError("control_stage must be commissioning or production")
        if not valid_workspace:
            raise ValueError("workspace_constraint does not match control_stage")
        if self.inputs_frozen_at_s > self.created_at_s + 1e-6:
            raise ValueError("inputs_frozen_at_s must not be after created_at_s")
        if self.valid_until_s <= self.created_at_s:
            raise ValueError("valid_until_s must be after created_at_s")
        for name in ("source_bucket_tip_stamp_s", "source_local_map_stamp_s"):
            stamp = getattr(self, name)
            if stamp <= 0.0 or stamp > self.inputs_frozen_at_s + 1e-6:
                raise ValueError(
                    "%s is inconsistent with inputs_frozen_at_s" % name
                )
            if self.inputs_frozen_at_s - stamp > 2.0:
                raise ValueError(
                    "%s is stale when planning inputs were frozen" % name
                )
        if current > self.valid_until_s:
            raise ValueError("trajectory snapshot expired")
        if current + _MAX_CLOCK_SKEW_S < self.created_at_s:
            raise ValueError("trajectory snapshot is from the future")


class EdgeFollowRuntimeFactory:
    """Create fresh Follow state while reusing validated deployment assets."""

    def __init__(
        self,
        *,
        machine_profile: Mapping[str, Any],
        kinematics: Any,
        policy: Any = None,
        controller: Any = None,
        controller_builder: Optional[Callable[[], Any]] = None,
        controller_backend: Optional[str] = None,
        mission: Mapping[str, Any],
        mission_sha256: str,
        runtime_type: Callable[..., Any] = EdgeFollowRuntime,
        action_slew_rate_per_s: Optional[float] = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(machine_profile, Mapping):
            raise ValueError("machine profile must be an object")
        if not isinstance(mission, Mapping):
            raise ValueError("mission must be an object")
        if machine_profile.get("machine_id") != "scale_excavator_v1":
            raise ValueError("unsupported machine profile")
        schema = machine_profile.get("observation_schema")
        if not isinstance(schema, Mapping):
            raise ValueError("machine profile observation_schema is missing")
        normalizers = schema.get("normalizers")
        if not isinstance(normalizers, Mapping):
            raise ValueError("machine profile normalizers are missing")
        _positive(
            "machine profile target_threshold",
            normalizers.get("target_threshold"),
        )
        _positive(
            "machine profile tube_radius",
            normalizers.get("tube_radius"),
        )
        MissionFollowLimits.from_mapping(mission)
        _text("mission_id", mission.get("mission_id"))
        self._machine_profile = machine_profile
        self._kinematics = kinematics
        controller_sources = sum(
            source is not None
            for source in (policy, controller, controller_builder)
        )
        if controller_sources != 1:
            raise ValueError(
                "exactly one controller builder, trajectory controller, "
                "or legacy policy is required"
            )
        if controller_builder is not None and not callable(controller_builder):
            raise ValueError("controller_builder must be callable")
        if controller_builder is not None:
            self._trajectory_controller_backend = _text(
                "controller_backend", controller_backend
            )
        elif controller is not None:
            descriptor = getattr(controller, "descriptor", None)
            self._trajectory_controller_backend = str(
                getattr(descriptor, "backend_id", "unknown")
            )
        else:
            self._trajectory_controller_backend = "legacy_policy"
        self._policy = policy
        self._controller = controller
        self._controller_builder = controller_builder
        self._mission = mission
        self._action_slew_rate_per_s = action_slew_rate_per_s
        if not callable(monotonic_clock):
            raise ValueError("monotonic_clock must be callable")
        self._monotonic_clock = monotonic_clock
        _sha256("mission_sha256", mission_sha256)
        self._runtime_type = runtime_type

    @property
    def trajectory_controller_backend(self) -> str:
        return self._trajectory_controller_backend

    @classmethod
    def from_config(cls, config: Any) -> "EdgeFollowRuntimeFactory":
        try:
            machine_profile = json.loads(
                config.machine_profile_path.read_text(encoding="utf-8")
            )
            mission_bytes = config.mission_path.read_bytes()
            mission = json.loads(mission_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "cannot read edge deployment artifact: %s" % exc
            ) from exc
        kinematics_type = _compatibility_dependency(
            "UrdfBucketTipKinematics", UrdfBucketTipKinematics
        )
        controller_builder_factory = _compatibility_dependency(
            "build_trajectory_controller_builder",
            build_trajectory_controller_builder,
        )
        return cls(
            machine_profile=machine_profile,
            kinematics=kinematics_type.from_path(config.urdf_path),
            controller_builder=_controller_builder_from_config(
                config,
                controller_builder_factory,
            ),
            controller_backend=config.trajectory_controller_backend,
            mission=mission,
            mission_sha256=hashlib.sha256(mission_bytes).hexdigest(),
            action_slew_rate_per_s=config.follow_action_slew_rate_per_s,
        )

    def create(self, snapshot: FollowTrajectorySnapshot) -> EdgeFollowRuntime:
        self._validate_mission(snapshot)
        normalizers = self._machine_profile["observation_schema"]["normalizers"]
        trajectory = {
            "schema_version": "trajectory_command.v1",
            "frame_id": snapshot.frame_id,
            "task_mode": snapshot.task_mode,
            "waypoints_base": [list(point) for point in snapshot.waypoints],
            "waypoint_count": len(snapshot.waypoints),
            "target_threshold": _positive(
                "machine profile target_threshold",
                normalizers.get("target_threshold"),
            ),
            "tube_radius": _positive(
                "machine profile tube_radius",
                normalizers.get("tube_radius"),
            ),
        }
        runtime_mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": snapshot.mission_id,
            "frame_id": snapshot.frame_id,
            "limits": {
                "waypoint_tolerance_m": snapshot.waypoint_tolerance_m,
                "waypoint_dwell_s": snapshot.waypoint_dwell_s,
                "tracking_timeout_s": snapshot.tracking_timeout_s,
            },
        }
        if snapshot.intermediate_waypoint_tolerance_m is not None:
            runtime_mission["limits"]["intermediate_waypoint_tolerance_m"] = (
                snapshot.intermediate_waypoint_tolerance_m
            )
        if self._controller_builder is not None:
            controller_arguments = {"controller": self._controller_builder()}
        elif self._controller is not None:
            controller_arguments = {"controller": self._controller}
        else:
            controller_arguments = {"policy": self._policy}
        return self._runtime_type(
            machine_profile=self._machine_profile,
            kinematics=self._kinematics,
            trajectory=trajectory,
            mission=runtime_mission,
            action_slew_rate_per_s=self._action_slew_rate_per_s,
            slew_started_monotonic_s=(
                self._monotonic_clock()
                if self._action_slew_rate_per_s is not None
                else None
            ),
            **controller_arguments,
        )

    def _validate_mission(self, snapshot: FollowTrajectorySnapshot) -> None:
        if self._mission.get("schema_version") != "excavation_mission.v1":
            raise ValueError(
                "mission schema_version must be excavation_mission.v1"
            )
        if self._mission.get("frame_id") != snapshot.frame_id:
            raise ValueError("trajectory frame does not match mission")
