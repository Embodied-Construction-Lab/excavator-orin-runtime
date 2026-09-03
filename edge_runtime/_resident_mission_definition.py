"""Immutable declarative Mission definitions and Behavior contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "resident_mission_definition.v1"
_MISSION_FIELDS = frozenset(
    {
        "schema_version",
        "mission_id",
        "entry_behavior",
        "cycle_behaviors",
        "act_policy_bindings",
    }
)
_PHASE_FIELDS = frozenset(
    {"stage_id", "behavior_id", "target_role", "max_steps"}
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TARGET_REFS = frozenset({"current_dig", "dump", "return_dig"})


@dataclass(frozen=True)
class BehaviorContract:
    behavior_id: str
    adapter: str
    trajectory_controller_backend: str | None = None
    fixed_action: str | None = None
    allow_deadzone_early_completion: bool = False


_BEHAVIOR_CONTRACTS = MappingProxyType(
    {
        "onnx_rl_tracking": BehaviorContract(
            behavior_id="onnx_rl_tracking",
            adapter="follow",
            trajectory_controller_backend="onnx_rl",
        ),
        "cartesian_p_tracking": BehaviorContract(
            behavior_id="cartesian_p_tracking",
            adapter="follow",
            trajectory_controller_backend="cartesian_p",
        ),
        "act_dig_lift": BehaviorContract(
            behavior_id="act_dig_lift",
            adapter="act",
            allow_deadzone_early_completion=True,
        ),
        "act_dig_transport_dump": BehaviorContract(
            behavior_id="act_dig_transport_dump",
            adapter="act",
            allow_deadzone_early_completion=False,
        ),
        "fixed_dig": BehaviorContract(
            behavior_id="fixed_dig",
            adapter="fixed_action",
            fixed_action="ExecuteDig",
        ),
        "fixed_dump": BehaviorContract(
            behavior_id="fixed_dump",
            adapter="fixed_action",
            fixed_action="ExecuteDump",
        ),
    }
)


@dataclass(frozen=True)
class MissionBehavior:
    stage_id: str
    behavior_id: str
    target_role: str | None
    max_steps: int | None

    @classmethod
    def from_mapping(cls, value: Any) -> "MissionBehavior":
        if not isinstance(value, Mapping) or set(value) != _PHASE_FIELDS:
            raise ValueError("mission behavior fields are invalid")
        stage_id = _identifier("stage_id", value["stage_id"])
        behavior_id = _identifier("behavior_id", value["behavior_id"])
        contract = behavior_contract(behavior_id)
        target_role = value["target_role"]
        max_steps = value["max_steps"]
        if contract.adapter == "follow":
            if target_role not in _TARGET_REFS or max_steps is not None:
                raise ValueError("tracking behavior target/max_steps are invalid")
        elif contract.adapter == "act":
            if target_role is not None:
                raise ValueError("ACT behavior must not declare a target_role")
            max_steps = _integer("max_steps", max_steps, 1, 2000)
        elif target_role is not None or max_steps is not None:
            raise ValueError("fixed behavior target/max_steps must be null")
        return cls(
            stage_id=stage_id,
            behavior_id=behavior_id,
            target_role=target_role,
            max_steps=max_steps,
        )

    @property
    def contract(self) -> BehaviorContract:
        return behavior_contract(self.behavior_id)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "behavior_id": self.behavior_id,
            "target_role": self.target_role,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class ResidentMissionDefinition:
    mission_id: str
    entry_behavior: MissionBehavior
    cycle_behaviors: tuple[MissionBehavior, ...]
    act_policy_bindings: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: Any) -> "ResidentMissionDefinition":
        if not isinstance(value, Mapping) or set(value) != _MISSION_FIELDS:
            raise ValueError("resident mission definition fields are invalid")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported resident mission definition schema")
        entry = MissionBehavior.from_mapping(value["entry_behavior"])
        raw_cycle = value["cycle_behaviors"]
        if not isinstance(raw_cycle, list) or not 1 <= len(raw_cycle) <= 12:
            raise ValueError("cycle_behaviors must contain between 1 and 12 items")
        cycle = tuple(MissionBehavior.from_mapping(item) for item in raw_cycle)
        phases = (entry, *cycle)
        if (
            entry.contract.adapter != "follow"
            or entry.target_role != "current_dig"
        ):
            raise ValueError("entry_behavior must track current_dig")
        if (
            cycle[-1].contract.adapter != "follow"
            or cycle[-1].target_role != "return_dig"
        ):
            raise ValueError("last cycle behavior must track return_dig")
        for phase in cycle[:-1]:
            if phase.target_role in {"current_dig", "return_dig"}:
                raise ValueError("only entry/final behavior may reference dig return")
        backends = {
            phase.contract.trajectory_controller_backend
            for phase in phases
            if phase.contract.trajectory_controller_backend is not None
        }
        if len(backends) != 1:
            raise ValueError("one Mission must use one trajectory controller backend")
        raw_bindings = value["act_policy_bindings"]
        if not isinstance(raw_bindings, Mapping):
            raise ValueError("act_policy_bindings must be an object")
        act_behavior_ids = {
            phase.behavior_id
            for phase in phases
            if phase.contract.adapter == "act"
        }
        if set(raw_bindings) != act_behavior_ids:
            raise ValueError("act_policy_bindings must exactly match ACT behaviors")
        if len(act_behavior_ids) > 1:
            raise ValueError("one Mission may bind only one resident ACT worker")
        bindings = {}
        for behavior_id, model_sha256 in raw_bindings.items():
            if not isinstance(model_sha256, str) or _SHA256.fullmatch(model_sha256) is None:
                raise ValueError("ACT policy model sha256 must be lowercase hexadecimal")
            bindings[behavior_id] = model_sha256
        return cls(
            mission_id=_identifier("mission_id", value["mission_id"]),
            entry_behavior=entry,
            cycle_behaviors=cycle,
            act_policy_bindings=MappingProxyType(bindings),
        )

    @property
    def behaviors(self) -> tuple[MissionBehavior, ...]:
        return (self.entry_behavior, *self.cycle_behaviors)

    @property
    def behavior_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.behavior_id for item in self.behaviors))

    @property
    def requires_act_worker(self) -> bool:
        return any(item.contract.adapter == "act" for item in self.behaviors)

    @property
    def act_worker_behavior_id(self) -> str | None:
        return next(iter(self.act_policy_bindings), None)

    @property
    def act_worker_model_sha256(self) -> str | None:
        behavior_id = self.act_worker_behavior_id
        return None if behavior_id is None else self.act_policy_bindings[behavior_id]

    @property
    def trajectory_controller_backend(self) -> str:
        for item in self.behaviors:
            backend = item.contract.trajectory_controller_backend
            if backend is not None:
                return backend
        raise RuntimeError("resident Mission has no trajectory controller")

    @property
    def target_roles(self) -> frozenset[str]:
        return frozenset(
            item.target_role
            for item in self.behaviors
            if item.target_role is not None
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "mission_id": self.mission_id,
            "act_policy_bindings": dict(self.act_policy_bindings),
            "entry_behavior": self.entry_behavior.to_mapping(),
            "cycle_behaviors": [
                behavior.to_mapping() for behavior in self.cycle_behaviors
            ],
        }

    @property
    def sha256(self) -> str:
        """Return the canonical identity of the complete Mission contract."""

        payload = json.dumps(
            self.to_mapping(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def behavior_contract(behavior_id: str) -> BehaviorContract:
    try:
        return _BEHAVIOR_CONTRACTS[behavior_id]
    except KeyError as exc:
        raise ValueError(f"unsupported resident behavior_id: {behavior_id}") from exc


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _integer(name: str, value: Any, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return value
