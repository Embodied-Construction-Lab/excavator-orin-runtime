"""Remote Follow RPC over length-prefixed TCP JSON.

This protocol owns behavior/session coordination only.  Physical actions still
flow exclusively through :class:`EdgeControlRunner` and the loopback
``ActionRelay``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional

from .follow import EdgeFollowRuntime
from .kinematics import UrdfBucketTipKinematics
from .onnx_policy import OnnxPolicy
from .remote_transport import (
    MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    RemoteBehaviorServer,
    receive_message,
    request_identity as _request_identity,
    send_message,
)
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
_DIGEST_FIELDS = _SNAPSHOT_FIELDS - {"trajectory_id", "trajectory_sha256"}
_MAX_CLOCK_SKEW_S = 0.5


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

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        now_s: float,
    ) -> "FollowTrajectorySnapshot":
        if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_FIELDS:
            raise ValueError("trajectory snapshot fields are invalid")
        snapshot = cls(
            trajectory_id=_text("trajectory_id", value["trajectory_id"]),
            trajectory_sha256=_sha256("trajectory_sha256", value["trajectory_sha256"]),
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
            execution_eligible=_boolean("execution_eligible", value["execution_eligible"]),
            source_bucket_tip_stamp_s=_finite(
                "source_bucket_tip_stamp_s",
                value["source_bucket_tip_stamp_s"],
            ),
            source_local_map_stamp_s=_finite(
                "source_local_map_stamp_s",
                value["source_local_map_stamp_s"],
            ),
            inputs_frozen_at_s=_finite("inputs_frozen_at_s", value["inputs_frozen_at_s"]),
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
        )
        snapshot._validate_for_execution(now_s=now_s)
        if snapshot.computed_sha256() != snapshot.trajectory_sha256:
            raise ValueError(
                "trajectory_sha256 does not match Trajectory Snapshot content"
            )
        return snapshot

    def computed_sha256(self) -> str:
        payload = {
            name: (
                [list(point) for point in self.waypoints]
                if name == "waypoints"
                else getattr(self, name)
            )
            for name in _DIGEST_FIELDS
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
                raise ValueError("%s is inconsistent with inputs_frozen_at_s" % name)
            if self.inputs_frozen_at_s - stamp > 2.0:
                raise ValueError("%s is stale when planning inputs were frozen" % name)
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
        policy: Any,
        mission: Mapping[str, Any],
        mission_sha256: str,
        runtime_type: Callable[..., Any] = EdgeFollowRuntime,
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
        self._policy = policy
        self._mission = mission
        _sha256("mission_sha256", mission_sha256)
        self._runtime_type = runtime_type

    @classmethod
    def from_config(cls, config: Any) -> "EdgeFollowRuntimeFactory":
        try:
            machine_profile = json.loads(
                config.machine_profile_path.read_text(encoding="utf-8")
            )
            mission_bytes = config.mission_path.read_bytes()
            mission = json.loads(mission_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read edge deployment artifact: %s" % exc) from exc
        return cls(
            machine_profile=machine_profile,
            kinematics=UrdfBucketTipKinematics.from_path(config.urdf_path),
            policy=OnnxPolicy(config.onnx_path),
            mission=mission,
            mission_sha256=hashlib.sha256(mission_bytes).hexdigest(),
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
        return self._runtime_type(
            machine_profile=self._machine_profile,
            kinematics=self._kinematics,
            policy=self._policy,
            trajectory=trajectory,
            mission=runtime_mission,
        )

    def _validate_mission(self, snapshot: FollowTrajectorySnapshot) -> None:
        if self._mission.get("schema_version") != "excavation_mission.v1":
            raise ValueError("mission schema_version must be excavation_mission.v1")
        if self._mission.get("frame_id") != snapshot.frame_id:
            raise ValueError("trajectory frame does not match mission")


@dataclass(frozen=True)
class _ActiveFollow:
    session_id: str
    request_id: str
    request_seq: int
    snapshot: FollowTrajectorySnapshot
    runner: Any
    event_sink: Callable[[Mapping[str, Any]], None]
    final_waypoint_index: int = 0
    final_distance_m: float = -1.0


@dataclass(frozen=True)
class _ActiveFixedAction:
    session_id: str
    request_id: str
    request_seq: int
    behavior: str
    runner: Any
    event_sink: Callable[[Mapping[str, Any]], None]
    final_step_index: int = 0
    final_step_label: str = ""
    final_max_error: float = 0.0


@dataclass(frozen=True)
class _SafetySnapshot:
    control_enabled: bool
    sensor_valid: bool
    stm32_alive: bool
    estop: bool
    fault_flags: tuple[str, ...]
    observed_monotonic_s: float


class EdgeBehaviorExecutor:
    """Serialize the one active Orin-local behavior and its state observations."""

    def __init__(
        self,
        *,
        runtime_factory: EdgeFollowRuntimeFactory,
        runner_factory: Callable[[EdgeFollowRuntime], Any],
        fixed_action_factory: Any = None,
        fixed_runner_factory: Optional[Callable[[Any, str], Any]] = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        action_stamp_clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        sender_constructed: bool = False,
        state_timeout_s: float = 0.3,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._runner_factory = runner_factory
        self._fixed_action_factory = fixed_action_factory
        self._fixed_runner_factory = fixed_runner_factory
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._action_stamp_clock = action_stamp_clock
        self._lock = threading.RLock()
        self._active: Optional[Any] = None
        self._event_sequence = 0
        self._last_request_sequences: dict[str, int] = {}
        self._latest_safety: Optional[_SafetySnapshot] = None
        self._last_action_datagrams = 0
        self._sender_constructed = bool(sender_constructed)
        self._state_timeout_s = _positive("state_timeout_s", state_timeout_s)

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active is not None

    def start(
        self,
        request: Mapping[str, Any],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        with self._lock:
            identity = _request_identity(request, expected_type="start_follow")
            if set(request) != {
                "schema_version",
                "type",
                "session_id",
                "seq",
                "request_id",
                "trajectory",
            }:
                self._reject(identity, event_sink, "BAD_REQUEST", "start_follow fields are invalid")
                return
            if not self._accept_request_sequence(identity, event_sink):
                return
            if self._active is not None:
                self._reject(identity, event_sink, "BUSY", "another behavior is active")
                return
            gate_reason = self._motion_gate_reason_locked(active=None)
            if gate_reason != "ready":
                self._reject(
                    identity,
                    event_sink,
                    "MOTION_NOT_READY",
                    gate_reason,
                )
                return
            try:
                snapshot = FollowTrajectorySnapshot.from_mapping(
                    request["trajectory"],
                    now_s=self._wall_clock(),
                )
                runtime = self._runtime_factory.create(snapshot)
                runner = self._runner_factory(runtime)
            except Exception as exc:
                self._reject(identity, event_sink, "INVALID_TRAJECTORY", str(exc))
                return
            self._active = _ActiveFollow(
                session_id=identity[0],
                request_id=identity[2],
                request_seq=identity[1],
                snapshot=snapshot,
                runner=runner,
                event_sink=event_sink,
            )
            self._emit(
                event_sink,
                "accepted",
                session_id=identity[0],
                request_id=identity[2],
                trajectory_id=snapshot.trajectory_id,
            )

    def handle(
        self,
        request: Mapping[str, Any],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        request_type = request.get("type") if isinstance(request, Mapping) else None
        if request_type == "start_follow":
            self.start(request, event_sink)
            return
        if request_type == "cancel_follow":
            self.cancel(request, event_sink)
            return
        if request_type == "start_fixed_action":
            self._start_fixed_action(request, event_sink)
            return
        if request_type == "cancel_fixed_action":
            self._cancel_fixed_action(request, event_sink)
            return
        identity = _request_identity(request)
        with self._lock:
            self._reject(
                identity,
                event_sink,
                "BAD_REQUEST",
                "unsupported behavior request type",
            )

    def _start_fixed_action(
        self,
        request: Mapping[str, Any],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        with self._lock:
            identity = _request_identity(
                request,
                expected_type="start_fixed_action",
            )
            if set(request) != {
                "schema_version",
                "type",
                "session_id",
                "seq",
                "request_id",
                "behavior",
            }:
                self._reject(
                    identity,
                    event_sink,
                    "BAD_REQUEST",
                    "start_fixed_action fields are invalid",
                )
                return
            if not self._accept_request_sequence(identity, event_sink):
                return
            if self._active is not None:
                self._reject(
                    identity,
                    event_sink,
                    "BUSY",
                    "another behavior is active",
                )
                return
            gate_reason = self._motion_gate_reason_locked(active=None)
            if gate_reason != "ready":
                self._reject(
                    identity,
                    event_sink,
                    "MOTION_NOT_READY",
                    gate_reason,
                )
                return
            behavior = request.get("behavior")
            if behavior not in {"ExecuteDig", "ExecuteDump"}:
                self._reject(
                    identity,
                    event_sink,
                    "INVALID_FIXED_ACTION",
                    "behavior must be ExecuteDig or ExecuteDump",
                )
                return
            if self._fixed_action_factory is None or self._fixed_runner_factory is None:
                self._reject(
                    identity,
                    event_sink,
                    "FIXED_ACTION_UNAVAILABLE",
                    "Orin fixed action runtime is unavailable",
                )
                return
            try:
                runtime = self._fixed_action_factory.create(behavior)
                runner = self._fixed_runner_factory(runtime, behavior)
            except Exception as exc:
                self._reject(
                    identity,
                    event_sink,
                    "INVALID_FIXED_ACTION",
                    str(exc),
                )
                return
            self._active = _ActiveFixedAction(
                session_id=identity[0],
                request_id=identity[2],
                request_seq=identity[1],
                behavior=behavior,
                runner=runner,
                event_sink=event_sink,
            )
            self._emit(
                event_sink,
                "accepted",
                session_id=identity[0],
                request_id=identity[2],
                behavior=behavior,
            )

    def _cancel_fixed_action(
        self,
        request: Mapping[str, Any],
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self._cancel_behavior(
            request,
            expected_type="cancel_fixed_action",
            expected_active_type=_ActiveFixedAction,
            event_sink=event_sink,
            message="fixed action cancelled by client",
        )

    def cancel(
        self,
        request: Mapping[str, Any],
        event_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self._cancel_behavior(
            request,
            expected_type="cancel_follow",
            expected_active_type=_ActiveFollow,
            event_sink=event_sink,
            message="Follow cancelled by client",
        )

    def _cancel_behavior(
        self,
        request: Mapping[str, Any],
        *,
        expected_type: str,
        expected_active_type: type,
        event_sink: Optional[Callable[[Mapping[str, Any]], None]],
        message: str,
    ) -> None:
        with self._lock:
            identity = _request_identity(request, expected_type=expected_type)
            active = self._active
            sink = event_sink or (
                active.event_sink if active is not None else None
            )
            if sink is None:
                raise ValueError("event_sink is required when no behavior is active")
            if set(request) != {
                "schema_version",
                "type",
                "session_id",
                "seq",
                "request_id",
            }:
                self._reject(
                    identity,
                    sink,
                    "BAD_REQUEST",
                    "%s fields are invalid" % expected_type,
                )
                return
            if not self._accept_request_sequence(identity, sink):
                return
            if active is None:
                self._reject(identity, sink, "NOT_ACTIVE", "no behavior is active")
                return
            if not isinstance(active, expected_active_type):
                self._reject(
                    identity,
                    sink,
                    "BEHAVIOR_MISMATCH",
                    "cancel request does not match the active behavior",
                )
                return
            if event_sink is not None and event_sink is not active.event_sink:
                self._reject(
                    identity,
                    sink,
                    "SESSION_MISMATCH",
                    "cancel connection does not own the active behavior",
                )
                return
            if identity[0] != active.session_id:
                self._reject(identity, sink, "SESSION_MISMATCH", "session_id is not active")
                return
            if identity[1] <= active.request_seq:
                self._reject(identity, sink, "OUT_OF_ORDER", "request seq is not increasing")
                return
            self._finish(
                outcome="CANCELLED",
                reason_code="CANCELLED",
                message=message,
            )

    def observe(self, machine_state: Mapping[str, Any]) -> None:
        with self._lock:
            observed_at = self._monotonic_clock()
            self._record_safety(machine_state, observed_at=observed_at)
            active = self._active
            if active is None:
                return
            try:
                step = active.runner.observe(
                    machine_state,
                    now_s=observed_at,
                    action_stamp_ms=self._action_stamp_clock(),
                )
            except Exception as exc:
                self._finish(
                    outcome="FAILED",
                    reason_code="EXECUTION_ERROR",
                    message=str(exc),
                )
                return
            if step is None:
                self._finish(
                    outcome="FAILED",
                    reason_code="STATE_REJECTED",
                    message="active behavior rejected the machine state",
                )
                return
            if isinstance(active, _ActiveFixedAction):
                self._observe_fixed_action(active, step)
                return
            self._active = replace(
                active,
                final_waypoint_index=step.waypoint_index,
                final_distance_m=step.waypoint_distance_m,
            )
            result = getattr(
                step,
                "result",
                "COMPLETED" if step.completed else "ACTIVE",
            )
            if result == "ACTIVE" and not step.completed:
                self._emit(
                    active.event_sink,
                    "feedback",
                    session_id=active.session_id,
                    request_id=active.request_id,
                    trajectory_id=active.snapshot.trajectory_id,
                    waypoint_index=step.waypoint_index,
                    waypoint_count=len(active.snapshot.waypoints),
                    distance_m=step.waypoint_distance_m,
                    elapsed_s=step.follow_elapsed_s,
                    bucket_tip_stamp_s=step.source_stamp_ms / 1000.0,
                    bucket_tip=list(step.bucket_tip_ros_m),
                    tracking_state="ACTIVE",
                    action_datagrams=_action_datagrams(active.runner),
                )
                return
            if result == "COMPLETED" or step.completed:
                self._finish(
                    outcome="SUCCEEDED",
                    reason_code="SUCCEEDED",
                    message="Follow completed",
                )
            elif result == "TIMEOUT":
                self._finish(
                    outcome="FAILED",
                    reason_code="TRACKING_TIMEOUT",
                    message="Follow tracking timeout",
                )
            else:
                self._finish(
                    outcome="FAILED",
                    reason_code="EXECUTION_ERROR",
                    message="Follow returned invalid terminal state",
                )

    def _observe_fixed_action(
        self,
        active: _ActiveFixedAction,
        step: Any,
    ) -> None:
        self._active = replace(
            active,
            final_step_index=step.step_index,
            final_step_label=step.step_label,
            final_max_error=step.max_error,
        )
        if step.result == "ACTIVE":
            self._emit(
                active.event_sink,
                "feedback",
                session_id=active.session_id,
                request_id=active.request_id,
                behavior=active.behavior,
                step_index=step.step_index,
                step_label=step.step_label,
                phase=step.phase,
                max_error=step.max_error,
                action_datagrams=_action_datagrams(active.runner),
            )
            return
        if step.result == "COMPLETED":
            self._finish(
                outcome="SUCCEEDED",
                reason_code=step.reason_code or "SEQUENCE_COMPLETED",
                message="%s completed" % active.behavior,
            )
        elif step.result == "TIMEOUT":
            self._finish(
                outcome="FAILED",
                reason_code=step.reason_code or "STEP_TIMEOUT",
                message="%s step timeout" % active.behavior,
            )
        else:
            self._finish(
                outcome="FAILED",
                reason_code=step.reason_code or "EXECUTION_ERROR",
                message="%s returned invalid terminal state" % active.behavior,
            )

    def close(self, *, emit_result: bool = True) -> None:
        with self._lock:
            if self._active is not None:
                self._finish(
                    outcome="FAILED",
                    reason_code="CONNECTION_CLOSED",
                    message="remote Follow connection closed",
                    emit_result=emit_result,
                )

    def watchdog(self) -> None:
        """Terminate an active Follow if the local motion gate has closed."""
        with self._lock:
            if self._active is None:
                return
            gate_reason = self._motion_gate_reason_locked(active=None)
            if gate_reason == "ready":
                return
            self._finish(
                outcome="FAILED",
                reason_code="MOTION_GATE_CLOSED",
                message=gate_reason,
            )

    def disconnect(
        self,
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        with self._lock:
            if (
                self._active is not None
                and self._active.event_sink is event_sink
            ):
                self._finish(
                    outcome="FAILED",
                    reason_code="CONNECTION_CLOSED",
                    message="remote Follow connection closed",
                    emit_result=False,
                )

    def status_event(self) -> dict[str, Any]:
        with self._lock:
            active = self._active
            safety = self._latest_safety
            fresh = (
                safety is not None
                and self._monotonic_clock() - safety.observed_monotonic_s
                <= self._state_timeout_s
            )
            control_enabled = safety.control_enabled if safety is not None else False
            sensor_valid = safety.sensor_valid if safety is not None else False
            stm32_alive = safety.stm32_alive if safety is not None else False
            estop = safety.estop if safety is not None else False
            fault_free = safety is not None and not safety.fault_flags
            gate_reason = self._motion_gate_reason_locked(active=active)
            return self._next_event(
                "status",
                session_id=active.session_id if active is not None else "server",
                request_id=active.request_id if active is not None else "status",
                state_fresh=fresh,
                control_enabled=control_enabled,
                sensor_valid=sensor_valid,
                stm32_alive=stm32_alive,
                estop=estop,
                fault_free=fault_free,
                quiescent=active is None,
                active_behavior=(
                    "Follow"
                    if isinstance(active, _ActiveFollow)
                    else (
                        active.behavior
                        if isinstance(active, _ActiveFixedAction)
                        else ""
                    )
                ),
                fixed_actions_available=self._fixed_action_factory is not None,
                fixed_actions_validated=bool(
                    self._fixed_action_factory is not None
                    and getattr(
                        getattr(self._fixed_action_factory, "profile", None),
                        "validation_status",
                        "",
                    )
                    == "field_validated"
                ),
                action_datagrams=(
                    _action_datagrams(active.runner)
                    if active is not None
                    else self._last_action_datagrams
                ),
                sender_constructed=self._sender_constructed,
                motion_gate_reason=gate_reason,
            )

    def _motion_gate_reason_locked(
        self,
        *,
        active: Optional[Any],
    ) -> str:
        safety = self._latest_safety
        if safety is None:
            return "state_unavailable"
        if (
            self._monotonic_clock() - safety.observed_monotonic_s
            > self._state_timeout_s
        ):
            return "state_stale"
        if safety.estop:
            return "estop"
        if not safety.stm32_alive:
            return "stm32_timeout"
        if not safety.sensor_valid:
            return "sensor_invalid"
        if safety.fault_flags:
            return "fault_flags"
        if not safety.control_enabled:
            return "control_disabled"
        if not self._sender_constructed:
            return "sender_unavailable"
        if active is not None:
            return "behavior_active"
        return "ready"

    def _finish(
        self,
        *,
        outcome: str,
        reason_code: str,
        message: str,
        emit_result: bool = True,
    ) -> None:
        active = self._active
        if active is None:
            return
        quiescent = False
        try:
            active.runner.close(action_stamp_ms=self._action_stamp_clock())
            quiescent = True
        except Exception as exc:
            quiescent = False
            message = "%s; terminal zero failed: %s" % (message, exc)
        finally:
            self._last_action_datagrams = _action_datagrams(active.runner)
            self._active = None
        if emit_result:
            fields = {
                "session_id": active.session_id,
                "request_id": active.request_id,
                "outcome": outcome,
                "reason_code": reason_code,
                "message": message,
                "quiescence_confirmed": quiescent,
                "action_datagrams": _action_datagrams(active.runner),
            }
            if isinstance(active, _ActiveFollow):
                fields.update(
                    {
                        "trajectory_id": active.snapshot.trajectory_id,
                        "final_waypoint_index": active.final_waypoint_index,
                        "final_distance_m": active.final_distance_m,
                    }
                )
            else:
                fields.update(
                    {
                        "behavior": active.behavior,
                        "final_step_index": active.final_step_index,
                        "final_step_label": active.final_step_label,
                        "final_max_error": active.final_max_error,
                    }
                )
            self._emit(active.event_sink, "result", **fields)

    def _reject(
        self,
        identity: tuple[str, int, str],
        event_sink: Callable[[Mapping[str, Any]], None],
        reason_code: str,
        message: str,
    ) -> None:
        self._emit(
            event_sink,
            "rejected",
            session_id=identity[0],
            request_id=identity[2],
            reason_code=reason_code,
            message=message,
        )

    def _accept_request_sequence(
        self,
        identity: tuple[str, int, str],
        event_sink: Callable[[Mapping[str, Any]], None],
    ) -> bool:
        last = self._last_request_sequences.get(identity[0])
        if last is not None and identity[1] <= last:
            self._reject(
                identity,
                event_sink,
                "OUT_OF_ORDER",
                "request seq is not increasing",
            )
            return False
        self._last_request_sequences = {
            **self._last_request_sequences,
            identity[0]: identity[1],
        }
        return True

    def _emit(
        self,
        event_sink: Callable[[Mapping[str, Any]], None],
        event_type: str,
        **fields: Any,
    ) -> None:
        event_sink(self._next_event(event_type, **fields))

    def _next_event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "schema_version": SCHEMA_VERSION,
            "type": event_type,
            "session_id": fields.pop("session_id"),
            "seq": self._event_sequence,
            "request_id": fields.pop("request_id"),
            **fields,
        }
        self._event_sequence += 1
        return event

    def _record_safety(
        self,
        machine_state: Mapping[str, Any],
        *,
        observed_at: float,
    ) -> None:
        safety = machine_state.get("safety")
        if not isinstance(safety, Mapping):
            return
        raw_faults = safety.get("fault_flags")
        faults = (
            tuple(str(value) for value in raw_faults)
            if isinstance(raw_faults, list)
            else ("fault_flags_invalid",)
        )
        self._latest_safety = _SafetySnapshot(
            control_enabled=safety.get("control_enabled") is True,
            sensor_valid=safety.get("sensor_valid") is True,
            stm32_alive=safety.get("stm32_alive") is True,
            estop=safety.get("estop") is True,
            fault_flags=faults,
            observed_monotonic_s=observed_at,
        )


def _action_datagrams(runner: Any) -> int:
    value = getattr(runner, "action_datagrams", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value
