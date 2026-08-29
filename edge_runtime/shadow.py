"""Shadow-mode edge inference integration and JSONL audit output."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .follow import EdgeFollowRuntime, EdgeFollowStep
from .kinematics import UrdfBucketTipKinematics
from .trajectory import validate_trajectory_mission
from .trajectory_controller import build_trajectory_controller


LOGGER = logging.getLogger("orin_edge_shadow")


def _runtime_trajectory_controller_backend(runtime: Any) -> str:
    direct = getattr(runtime, "trajectory_controller_backend", None)
    if isinstance(direct, str) and direct.strip():
        return direct
    controller = getattr(runtime, "_controller", None)
    descriptor = getattr(controller, "descriptor", None)
    backend = getattr(descriptor, "backend_id", None)
    if isinstance(backend, str) and backend.strip():
        return backend
    return "unknown"


@dataclass(frozen=True)
class RemoteBehaviorConfig:
    bind_host: str
    bind_port: int
    allowed_client_host: str
    status_hz: float
    status_timeout_s: float


@dataclass(frozen=True)
class EdgeRuntimeConfig:
    mode: str
    action_transport: str
    machine_profile_path: Path
    urdf_path: Path
    onnx_path: Path | None
    trajectory_controller_backend: str
    trajectory_path: Optional[Path]
    mission_path: Path
    fixed_action_profile_path: Optional[Path]
    audit_path: Path
    action_valid_for_ms: int
    follow_action_slew_rate_per_s: Optional[float] = None
    follow_action_startup_slew_rate_per_s: Optional[float] = None
    manual_action_deadzone_path: Optional[Path] = None
    remote_behavior: Optional[RemoteBehaviorConfig] = None


def load_edge_runtime_config(path: Path) -> EdgeRuntimeConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read edge shadow config %s: %s" % (path, exc)) from exc
    common = {
        "schema_version",
        "mode",
        "action_transport",
        "machine_profile_path",
        "urdf_path",
        "mission_path",
        "audit_path",
        "action_valid_for_ms",
    }
    optional = {
        "follow_action_slew_rate_per_s",
        "follow_action_startup_slew_rate_per_s",
        "trajectory_controller_backend",
        "onnx_path",
        "manual_action_deadzone_path",
    }
    if (
        not isinstance(value, dict)
        or "schema_version" not in value
        or "mode" not in value
    ):
        raise ValueError("edge shadow config fields are invalid")
    if value["schema_version"] != "orin_edge_runtime.v1":
        raise ValueError("edge shadow schema_version is invalid")
    mode = value["mode"]
    if mode not in {"shadow", "control", "remote_control"}:
        raise ValueError(
            "edge runtime mode must be shadow, control, or remote_control"
        )
    trajectory_controller_backend = value.get(
        "trajectory_controller_backend", "onnx_rl"
    )
    if trajectory_controller_backend not in {"onnx_rl", "cartesian_p"}:
        raise ValueError(
            "trajectory_controller_backend must be onnx_rl or cartesian_p"
        )
    controller_required = (
        {"onnx_path"}
        if trajectory_controller_backend == "onnx_rl"
        else set()
    )
    if trajectory_controller_backend == "onnx_rl" and (
        not isinstance(value.get("onnx_path"), str)
        or not value["onnx_path"].strip()
    ):
        raise ValueError("onnx_path is required for onnx_rl")
    if mode in ("shadow", "control", "remote_control"):
        required = common | controller_required | (
            {"trajectory_path"}
            if mode in ("shadow", "control")
            else {"remote_behavior", "fixed_action_profile_path"}
        )
        legacy_required = required - {"action_transport"}
        if legacy_required <= set(value) <= legacy_required | optional:
            value = dict(value)
            value["action_transport"] = "loopback_udp"
    action_transport = value.get("action_transport")
    if action_transport not in {"loopback_udp", "resident_sink"}:
        raise ValueError("edge action_transport must be loopback_udp or resident_sink")
    if mode in ("shadow", "control"):
        required = common | controller_required | {"trajectory_path"}
        if not required <= set(value) <= required | optional:
            raise ValueError("edge shadow config fields are invalid")
        remote_behavior = None
    elif mode == "remote_control":
        required = (
            common
            | controller_required
            | {"remote_behavior", "fixed_action_profile_path"}
        )
        if not required <= set(value) <= required | optional:
            raise ValueError("remote edge config fields are invalid")
        remote_behavior = _remote_behavior_config(value["remote_behavior"])
    else:
        raise ValueError(
            "edge runtime mode must be shadow, control, or remote_control"
        )
    action_valid_for_ms = value["action_valid_for_ms"]
    if (
        isinstance(action_valid_for_ms, bool)
        or not isinstance(action_valid_for_ms, int)
        or action_valid_for_ms <= 0
    ):
        raise ValueError("edge action_valid_for_ms must be a positive integer")
    follow_action_slew_rate_per_s = value.get(
        "follow_action_slew_rate_per_s"
    )
    if follow_action_slew_rate_per_s is not None:
        if (
            isinstance(follow_action_slew_rate_per_s, bool)
            or not isinstance(follow_action_slew_rate_per_s, (int, float))
            or not math.isfinite(float(follow_action_slew_rate_per_s))
            or float(follow_action_slew_rate_per_s) <= 0.0
        ):
            raise ValueError(
                "follow_action_slew_rate_per_s must be finite and positive"
            )
        follow_action_slew_rate_per_s = float(
            follow_action_slew_rate_per_s
        )
    follow_action_startup_slew_rate_per_s = value.get(
        "follow_action_startup_slew_rate_per_s"
    )
    if follow_action_startup_slew_rate_per_s is not None:
        if (
            isinstance(follow_action_startup_slew_rate_per_s, bool)
            or not isinstance(
                follow_action_startup_slew_rate_per_s,
                (int, float),
            )
            or not math.isfinite(
                float(follow_action_startup_slew_rate_per_s)
            )
            or float(follow_action_startup_slew_rate_per_s) <= 0.0
        ):
            raise ValueError(
                "follow_action_startup_slew_rate_per_s must be finite and positive"
            )
        if follow_action_slew_rate_per_s is None:
            raise ValueError(
                "follow_action_startup_slew_rate_per_s requires "
                "follow_action_slew_rate_per_s"
            )
        follow_action_startup_slew_rate_per_s = float(
            follow_action_startup_slew_rate_per_s
        )
    root = config_path.parent
    return EdgeRuntimeConfig(
        mode=mode,
        action_transport=action_transport,
        machine_profile_path=_relative_path(root, value["machine_profile_path"]),
        urdf_path=_relative_path(root, value["urdf_path"]),
        onnx_path=(
            _relative_path(root, value["onnx_path"])
            if "onnx_path" in value
            else None
        ),
        trajectory_controller_backend=trajectory_controller_backend,
        trajectory_path=(
            _relative_path(root, value["trajectory_path"])
            if mode != "remote_control"
            else None
        ),
        mission_path=_relative_path(root, value["mission_path"]),
        fixed_action_profile_path=(
            _relative_path(root, value["fixed_action_profile_path"])
            if mode == "remote_control"
            else None
        ),
        audit_path=_relative_path(root, value["audit_path"]),
        action_valid_for_ms=action_valid_for_ms,
        follow_action_slew_rate_per_s=follow_action_slew_rate_per_s,
        follow_action_startup_slew_rate_per_s=(
            follow_action_startup_slew_rate_per_s
        ),
        manual_action_deadzone_path=(
            _relative_path(root, value["manual_action_deadzone_path"])
            if "manual_action_deadzone_path" in value
            else None
        ),
        remote_behavior=remote_behavior,
    )


def build_edge_follow_runtime(config: EdgeRuntimeConfig) -> EdgeFollowRuntime:
    if config.trajectory_path is None:
        raise ValueError("static edge runtime requires trajectory_path")
    try:
        machine_profile = json.loads(
            config.machine_profile_path.read_text(encoding="utf-8")
        )
        trajectory = json.loads(config.trajectory_path.read_text(encoding="utf-8"))
        mission_bytes = config.mission_path.read_bytes()
        mission = json.loads(mission_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read edge deployment artifact: %s" % exc) from exc
    validate_trajectory_mission(
        trajectory,
        mission,
        mission_sha256=hashlib.sha256(mission_bytes).hexdigest(),
    )
    return EdgeFollowRuntime(
        machine_profile=machine_profile,
        kinematics=UrdfBucketTipKinematics.from_path(config.urdf_path),
        controller=build_trajectory_controller(
            config.trajectory_controller_backend,
            onnx_path=config.onnx_path,
        ),
        trajectory=trajectory,
        mission=mission,
        action_slew_rate_per_s=config.follow_action_slew_rate_per_s,
        action_startup_slew_rate_per_s=(
            config.follow_action_startup_slew_rate_per_s
        ),
    )


def build_edge_shadow_observer(config_path: Path) -> "EdgeShadowObserver":
    config = load_edge_runtime_config(config_path)
    if config.mode != "shadow":
        raise ValueError("edge shadow observer requires mode=shadow")
    return EdgeShadowObserver(
        runtime=build_edge_follow_runtime(config),
        audit_path=config.audit_path,
    )


class EdgeShadowObserver:
    """Observe state and record local inference; intentionally has no command sink."""

    def __init__(self, *, runtime: EdgeFollowRuntime, audit_path: Path) -> None:
        self._runtime = runtime
        self._audit_path = Path(audit_path)
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_handle = self._audit_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._consecutive_rejections = 0
        self._trajectory_controller_backend = (
            _runtime_trajectory_controller_backend(runtime)
        )

    def observe(
        self,
        machine_state: Mapping[str, Any],
        *,
        now_s: Optional[float] = None,
    ) -> Optional[EdgeFollowStep]:
        started = time.perf_counter()
        runtime_time = time.monotonic() if now_s is None else float(now_s)
        try:
            step = self._runtime.step(machine_state, now_s=runtime_time)
        except Exception as exc:
            self._consecutive_rejections += 1
            loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
            record = {
                "schema_version": "orin_edge_shadow_audit.v1",
                "mode": "shadow",
                "status": "rejected",
                "source_seq": machine_state.get("seq"),
                "source_stamp_ms": machine_state.get("stamp_ms"),
                "reason": str(exc),
                "exception_type": type(exc).__name__,
                "trajectory_controller_backend": (
                    self._trajectory_controller_backend
                ),
                "consecutive_rejections": self._consecutive_rejections,
                "runtime_monotonic_s": runtime_time,
                "inference_elapsed_ms": loop_elapsed_ms,
                "loop_elapsed_ms": loop_elapsed_ms,
            }
            self._append(record)
            LOGGER.warning(
                "edge shadow rejected state seq=%s: %s",
                machine_state.get("seq"),
                exc,
            )
            return None

        self._consecutive_rejections = 0
        step_backend = getattr(
            step,
            "trajectory_controller_backend",
            self._trajectory_controller_backend,
        )
        if isinstance(step_backend, str) and step_backend.strip():
            self._trajectory_controller_backend = step_backend
        loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
        record = asdict(step)
        record.update(
            {
                "schema_version": "orin_edge_shadow_audit.v1",
                "mode": "shadow",
                "status": step.result,
                "trajectory_controller_backend": (
                    self._trajectory_controller_backend
                ),
                "consecutive_rejections": self._consecutive_rejections,
                "runtime_monotonic_s": runtime_time,
                "inference_elapsed_ms": loop_elapsed_ms,
                "loop_elapsed_ms": loop_elapsed_ms,
            }
        )
        self._append(record)
        return step

    def _append(self, record: Mapping[str, Any]) -> None:
        self._audit_handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    def close(self) -> None:
        self._audit_handle.close()


def _relative_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("edge shadow path must be a non-empty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _remote_behavior_config(value: Any) -> RemoteBehaviorConfig:
    required = {
        "bind_host",
        "bind_port",
        "allowed_client_host",
        "status_hz",
        "status_timeout_s",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("remote_behavior fields are invalid")
    for name in ("bind_host", "allowed_client_host"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError("remote_behavior.%s must be non-empty" % name)
    bind_port = value["bind_port"]
    if (
        isinstance(bind_port, bool)
        or not isinstance(bind_port, int)
        or bind_port <= 0
        or bind_port > 65535
    ):
        raise ValueError("remote_behavior.bind_port is invalid")
    numeric = {}
    for name in ("status_hz", "status_timeout_s"):
        item = value[name]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
        ):
            raise ValueError("remote_behavior.%s must be positive" % name)
        numeric[name] = float(item)
    return RemoteBehaviorConfig(
        bind_host=value["bind_host"],
        bind_port=bind_port,
        allowed_client_host=value["allowed_client_host"],
        status_hz=numeric["status_hz"],
        status_timeout_s=numeric["status_timeout_s"],
    )
