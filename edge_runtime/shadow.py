"""Shadow-mode edge inference integration and JSONL audit output."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .follow import EdgeFollowRuntime, EdgeFollowStep
from .kinematics import UrdfBucketTipKinematics
from .onnx_policy import OnnxPolicy
from .trajectory import validate_trajectory_mission


LOGGER = logging.getLogger("orin_edge_shadow")


@dataclass(frozen=True)
class EdgeRuntimeConfig:
    mode: str
    machine_profile_path: Path
    urdf_path: Path
    onnx_path: Path
    trajectory_path: Path
    mission_path: Path
    audit_path: Path
    action_valid_for_ms: int


def load_edge_runtime_config(path: Path) -> EdgeRuntimeConfig:
    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read edge shadow config %s: %s" % (path, exc)) from exc
    required = {
        "schema_version",
        "mode",
        "machine_profile_path",
        "urdf_path",
        "onnx_path",
        "trajectory_path",
        "mission_path",
        "audit_path",
        "action_valid_for_ms",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("edge shadow config fields are invalid")
    if value["schema_version"] != "orin_edge_runtime.v1":
        raise ValueError("edge shadow schema_version is invalid")
    if value["mode"] not in ("shadow", "control"):
        raise ValueError("edge runtime mode must be shadow or control")
    action_valid_for_ms = value["action_valid_for_ms"]
    if (
        isinstance(action_valid_for_ms, bool)
        or not isinstance(action_valid_for_ms, int)
        or action_valid_for_ms <= 0
    ):
        raise ValueError("edge action_valid_for_ms must be a positive integer")
    root = config_path.parent
    return EdgeRuntimeConfig(
        mode=value["mode"],
        machine_profile_path=_relative_path(root, value["machine_profile_path"]),
        urdf_path=_relative_path(root, value["urdf_path"]),
        onnx_path=_relative_path(root, value["onnx_path"]),
        trajectory_path=_relative_path(root, value["trajectory_path"]),
        mission_path=_relative_path(root, value["mission_path"]),
        audit_path=_relative_path(root, value["audit_path"]),
        action_valid_for_ms=action_valid_for_ms,
    )


def build_edge_follow_runtime(config: EdgeRuntimeConfig) -> EdgeFollowRuntime:
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
        policy=OnnxPolicy(config.onnx_path),
        trajectory=trajectory,
        mission=mission,
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
                "consecutive_rejections": self._consecutive_rejections,
                "runtime_monotonic_s": runtime_time,
                "inference_elapsed_ms": loop_elapsed_ms,
                "loop_elapsed_ms": loop_elapsed_ms,
            }
            self._append(record)
            LOGGER.warning("edge shadow rejected state seq=%s: %s", machine_state.get("seq"), exc)
            return None

        self._consecutive_rejections = 0
        loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
        record = asdict(step)
        record.update(
            {
                "schema_version": "orin_edge_shadow_audit.v1",
                "mode": "shadow",
                "status": step.result,
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
