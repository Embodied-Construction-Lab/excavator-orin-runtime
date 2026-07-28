"""Orin-local closed-loop ExecuteDig and ExecuteDump behaviors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from .actions import ACTION_ORDER, physical_velocity_from_normalized
from .observation import normalize_position


_ZERO_ACTION = (0.0, 0.0, 0.0, 0.0)
_PHASES = ("dig", "dump")


@dataclass(frozen=True)
class FixedActionController:
    kp: float
    min_action: float
    max_action: float
    tolerance: float
    step_timeout_s: float
    hold_s: float


@dataclass(frozen=True)
class FixedActionStep:
    step_id: str
    label: str
    delta_normalized_qpos: Tuple[float, float, float, float]


@dataclass(frozen=True)
class FixedActionEnvelope:
    normalized_actuator_position: Mapping[str, Tuple[float, float]]
    bucket_pitch_deg: Tuple[float, float]
    swing_rad: Tuple[float, float]


@dataclass(frozen=True)
class FixedActionProfile:
    profile_id: str
    machine_id: str
    validation_status: str
    machine_profile_sha256: str
    urdf_sha256: str
    controller: FixedActionController
    start_envelopes: Mapping[str, FixedActionEnvelope]
    actions: Mapping[str, Tuple[FixedActionStep, ...]]
    sha256: str = ""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        sha256: str = "",
    ) -> "FixedActionProfile":
        required = {
            "schema_version",
            "profile_id",
            "machine_id",
            "action_order",
            "validation_status",
            "validation_evidence",
            "machine_profile_sha256",
            "urdf_sha256",
            "controller",
            "start_envelopes",
            "actions",
        }
        _require_fields("fixed action profile", value, required)
        if value["schema_version"] != "fixed_action_profile.v1":
            raise ValueError("fixed action profile schema_version is invalid")
        if tuple(value["action_order"]) != ACTION_ORDER:
            raise ValueError("fixed action action_order is invalid")
        validation_status = _text("validation_status", value["validation_status"])
        if validation_status not in {"candidate", "field_validated"}:
            raise ValueError("fixed action validation_status is invalid")
        if validation_status == "candidate" and value["validation_evidence"] is not None:
            raise ValueError("candidate fixed action must not declare validation evidence")
        actions_value = _mapping("actions", value["actions"])
        envelopes_value = _mapping("start_envelopes", value["start_envelopes"])
        _require_fields("actions", actions_value, set(_PHASES))
        _require_fields("start_envelopes", envelopes_value, set(_PHASES))
        return cls(
            profile_id=_text("profile_id", value["profile_id"]),
            machine_id=_text("machine_id", value["machine_id"]),
            validation_status=validation_status,
            machine_profile_sha256=_digest(
                "machine_profile_sha256", value["machine_profile_sha256"]
            ),
            urdf_sha256=_digest("urdf_sha256", value["urdf_sha256"]),
            controller=_controller(value["controller"]),
            start_envelopes=MappingProxyType(
                {
                    phase: _envelope(phase, envelopes_value[phase])
                    for phase in _PHASES
                }
            ),
            actions=MappingProxyType(
                {
                    phase: _steps(phase, actions_value[phase])
                    for phase in _PHASES
                }
            ),
            sha256=sha256,
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        machine_profile_path: Path,
        urdf_path: Path,
    ) -> "FixedActionProfile":
        try:
            raw = Path(path).read_bytes()
            value = json.loads(raw.decode("utf-8"))
            machine_profile_raw = Path(machine_profile_path).read_bytes()
            machine_profile = json.loads(machine_profile_raw.decode("utf-8"))
            urdf_raw = Path(urdf_path).read_bytes()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read fixed action deployment artifact: %s" % exc) from exc
        profile = cls.from_mapping(
            _mapping("fixed action profile", value),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        if machine_profile.get("machine_id") != profile.machine_id:
            raise ValueError("fixed action machine_id does not match machine profile")
        if tuple(machine_profile.get("action_order", ())) != ACTION_ORDER:
            raise ValueError("machine profile action_order is invalid")
        if hashlib.sha256(machine_profile_raw).hexdigest() != profile.machine_profile_sha256:
            raise ValueError("fixed action machine_profile_sha256 is stale")
        if hashlib.sha256(urdf_raw).hexdigest() != profile.urdf_sha256:
            raise ValueError("fixed action urdf_sha256 is stale")
        return profile


@dataclass(frozen=True)
class FixedActionRuntimeStep:
    phase: str
    step_index: int
    step_label: str
    max_error: float
    normalized_action: Tuple[float, float, float, float]
    physical_action: Tuple[float, float, float, float]
    result: str
    reason_code: str = ""


class FixedActionRuntime:
    """Track a relative normalized actuator sequence from local Machine State."""

    def __init__(
        self,
        *,
        profile: FixedActionProfile,
        machine_profile: Mapping[str, Any],
        phase: str,
    ) -> None:
        if phase not in _PHASES:
            raise ValueError("fixed action phase must be dig or dump")
        if profile.machine_id != machine_profile.get("machine_id"):
            raise ValueError("fixed action profile does not match machine profile")
        if tuple(machine_profile.get("action_order", ())) != ACTION_ORDER:
            raise ValueError("machine profile action_order is invalid")
        self._profile = profile
        self._machine_profile = machine_profile
        self._phase = phase
        self._steps = profile.actions[phase]
        self._step_index = 0
        self._step_started_at_s: float | None = None
        self._step_start_qpos: Tuple[float, float, float, float] | None = None
        self._hold_until_s: float | None = None
        self._terminal_result = ""
        self._terminal_reason = ""
        self._start_checked = False

    def step(
        self,
        machine_state: Mapping[str, Any],
        *,
        now_s: float,
    ) -> FixedActionRuntimeStep:
        current = self._current_qpos(machine_state)
        if not self._start_checked:
            self._check_start_envelope(current, machine_state)
            self._start_checked = True
        if self._terminal_result:
            return self._terminal(self._terminal_result, self._terminal_reason)
        if self._hold_until_s is not None:
            if now_s < self._hold_until_s:
                return self._active("hold", 0.0, _ZERO_ACTION, _ZERO_ACTION)
            self._advance()
            if self._step_index >= len(self._steps):
                self._terminal_result = "COMPLETED"
                return self._terminal("COMPLETED", "SEQUENCE_COMPLETED")

        if self._step_started_at_s is None:
            self._step_started_at_s = float(now_s)
            self._step_start_qpos = current
        step = self._steps[self._step_index]
        start = self._step_start_qpos or current
        target = tuple(
            _clamp(start[index] + step.delta_normalized_qpos[index])
            for index in range(4)
        )
        error = tuple(
            0.0
            if abs(step.delta_normalized_qpos[index]) <= 1e-4
            else target[index] - current[index]
            for index in range(4)
        )
        max_error = max(abs(value) for value in error)
        controller = self._profile.controller
        if float(now_s) - self._step_started_at_s >= controller.step_timeout_s:
            self._terminal_result = "TIMEOUT"
            self._terminal_reason = "STEP_TIMEOUT"
            return self._terminal("TIMEOUT", "STEP_TIMEOUT", max_error=max_error)
        if max_error <= controller.tolerance:
            self._hold_until_s = float(now_s) + controller.hold_s
            return self._active("hold", max_error, _ZERO_ACTION, _ZERO_ACTION)
        normalized = tuple(self._servo_axis(value) for value in error)
        physical = physical_velocity_from_normalized(
            normalized,
            self._machine_profile,
        )
        return self._active("running", max_error, normalized, physical)

    def _current_qpos(
        self, machine_state: Mapping[str, Any]
    ) -> Tuple[float, float, float, float]:
        actuator_state = _mapping("actuator_state", machine_state.get("actuator_state"))
        actuators = _mapping("machine profile actuators", self._machine_profile.get("actuators"))
        values = tuple(
            normalize_position(
                _finite(
                    "%s.position_m" % name,
                    _mapping(name, actuator_state.get(name)).get("position_m"),
                ),
                _mapping(name, actuators.get(name)),
            )
            for name in ("boom", "stick", "bucket")
        )
        return values + (0.0,)

    def _check_start_envelope(
        self,
        current: Tuple[float, float, float, float],
        machine_state: Mapping[str, Any],
    ) -> None:
        envelope = self._profile.start_envelopes[self._phase]
        for index, name in enumerate(("boom", "stick", "bucket")):
            lower, upper = envelope.normalized_actuator_position[name]
            if current[index] < lower or current[index] > upper:
                raise ValueError("%s outside fixed action start envelope" % name)
        swing_state = _mapping(
            "swing actuator state",
            _mapping("actuator_state", machine_state.get("actuator_state")).get("swing"),
        )
        swing = _finite("swing.position_rad", swing_state.get("position_rad"))
        if not envelope.swing_rad[0] <= swing <= envelope.swing_rad[1]:
            raise ValueError("swing outside fixed action start envelope")

    def _servo_axis(self, error: float) -> float:
        controller = self._profile.controller
        if abs(error) <= controller.tolerance:
            return 0.0
        magnitude = _clamp(
            abs(controller.kp * error),
            controller.min_action,
            controller.max_action,
        )
        return magnitude if error >= 0.0 else -magnitude

    def _advance(self) -> None:
        self._step_index += 1
        self._step_started_at_s = None
        self._step_start_qpos = None
        self._hold_until_s = None

    def _active(
        self,
        runtime_phase: str,
        max_error: float,
        normalized: Sequence[float],
        physical: Sequence[float],
    ) -> FixedActionRuntimeStep:
        step = self._steps[self._step_index]
        return FixedActionRuntimeStep(
            phase=runtime_phase,
            step_index=self._step_index,
            step_label=step.label,
            max_error=float(max_error),
            normalized_action=tuple(float(value) for value in normalized),
            physical_action=tuple(float(value) for value in physical),
            result="ACTIVE",
        )

    def _terminal(
        self,
        result: str,
        reason: str,
        *,
        max_error: float = 0.0,
    ) -> FixedActionRuntimeStep:
        index = min(self._step_index, len(self._steps) - 1)
        return FixedActionRuntimeStep(
            phase="done" if result == "COMPLETED" else "failed",
            step_index=index,
            step_label=self._steps[index].label,
            max_error=float(max_error),
            normalized_action=_ZERO_ACTION,
            physical_action=_ZERO_ACTION,
            result=result,
            reason_code=reason,
        )


class FixedActionRuntimeFactory:
    def __init__(
        self,
        *,
        profile: FixedActionProfile,
        machine_profile: Mapping[str, Any],
    ) -> None:
        self.profile = profile
        self._machine_profile = machine_profile

    @classmethod
    def from_config(cls, config: Any) -> "FixedActionRuntimeFactory":
        try:
            machine_profile = json.loads(
                config.machine_profile_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read machine profile for fixed actions: %s" % exc) from exc
        if config.fixed_action_profile_path is None:
            raise ValueError("remote control requires fixed_action_profile_path")
        return cls(
            profile=FixedActionProfile.from_path(
                config.fixed_action_profile_path,
                machine_profile_path=config.machine_profile_path,
                urdf_path=config.urdf_path,
            ),
            machine_profile=machine_profile,
        )

    def create(self, behavior: str) -> FixedActionRuntime:
        phase = {"ExecuteDig": "dig", "ExecuteDump": "dump"}.get(behavior)
        if phase is None:
            raise ValueError("unsupported fixed action behavior")
        return FixedActionRuntime(
            profile=self.profile,
            machine_profile=self._machine_profile,
            phase=phase,
        )


def _controller(value: Any) -> FixedActionController:
    mapping = _mapping("controller", value)
    fields = {"kp", "min_action", "max_action", "tolerance", "step_timeout_s", "hold_s"}
    _require_fields("controller", mapping, fields)
    converted = {name: _finite("controller.%s" % name, mapping[name]) for name in fields}
    if not 0.0 < converted["kp"] <= 100.0:
        raise ValueError("fixed action kp is invalid")
    if not 0.0 <= converted["min_action"] <= converted["max_action"] <= 1.0:
        raise ValueError("fixed action command bounds are invalid")
    if not 0.0 < converted["tolerance"] <= 1.0:
        raise ValueError("fixed action tolerance is invalid")
    if not 0.0 < converted["step_timeout_s"] <= 60.0:
        raise ValueError("fixed action step timeout is invalid")
    if not 0.0 <= converted["hold_s"] <= 10.0:
        raise ValueError("fixed action hold duration is invalid")
    return FixedActionController(**converted)


def _envelope(phase: str, value: Any) -> FixedActionEnvelope:
    mapping = _mapping("start_envelopes.%s" % phase, value)
    fields = {"normalized_actuator_position", "bucket_pitch_deg", "swing_rad"}
    _require_fields("start_envelopes.%s" % phase, mapping, fields)
    normalized = _mapping(
        "start_envelopes.%s.normalized_actuator_position" % phase,
        mapping["normalized_actuator_position"],
    )
    _require_fields("normalized actuator envelope", normalized, {"boom", "stick", "bucket"})
    return FixedActionEnvelope(
        normalized_actuator_position=MappingProxyType(
            {
                name: _range("%s.%s" % (phase, name), normalized[name], -1.0, 1.0)
                for name in ("boom", "stick", "bucket")
            }
        ),
        bucket_pitch_deg=_range(
            "%s.bucket_pitch_deg" % phase,
            mapping["bucket_pitch_deg"],
            -180.0,
            180.0,
        ),
        swing_rad=_range(
            "%s.swing_rad" % phase,
            mapping["swing_rad"],
            -math.tau,
            math.tau,
        ),
    )


def _steps(phase: str, value: Any) -> Tuple[FixedActionStep, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("actions.%s must be a non-empty array" % phase)
    result = []
    identifiers = set()
    for index, item in enumerate(value):
        mapping = _mapping("actions.%s[%d]" % (phase, index), item)
        _require_fields(
            "fixed action step",
            mapping,
            {"step_id", "label", "delta_by_actuator"},
        )
        step_id = _text("step_id", mapping["step_id"])
        if step_id in identifiers:
            raise ValueError("fixed action step_id must be unique")
        identifiers.add(step_id)
        deltas = _mapping("delta_by_actuator", mapping["delta_by_actuator"])
        _require_fields("delta_by_actuator", deltas, set(ACTION_ORDER))
        converted = tuple(
            _finite("%s delta" % name, deltas[name]) for name in ACTION_ORDER
        )
        if any(abs(value) > 2.0 for value in converted):
            raise ValueError("fixed action delta must be within [-2, 2]")
        if not any(abs(value) > 1e-9 for value in converted):
            raise ValueError("fixed action step must not be all zero")
        if abs(converted[3]) > 1e-9:
            raise ValueError("fixed action profile does not support swing deltas")
        result.append(
            FixedActionStep(
                step_id=step_id,
                label=_text("label", mapping["label"]),
                delta_normalized_qpos=converted,
            )
        )
    return tuple(result)


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % name)
    return value


def _require_fields(
    name: str,
    value: Mapping[str, Any],
    expected: set[str],
) -> None:
    if set(value) != expected:
        raise ValueError("%s fields are invalid" % name)


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be a non-empty string" % name)
    return value


def _digest(name: str, value: Any) -> str:
    text = _text(name, value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("%s must be a lowercase SHA-256" % name)
    return text


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be finite" % name)
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("%s must be finite" % name)
    return converted


def _range(
    name: str,
    value: Any,
    minimum: float,
    maximum: float,
) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("%s must contain two values" % name)
    lower, upper = (_finite(name, item) for item in value)
    if lower < minimum or upper > maximum or lower > upper:
        raise ValueError("%s is invalid" % name)
    return lower, upper


def _clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
