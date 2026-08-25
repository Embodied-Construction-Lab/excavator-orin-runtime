from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_runtime.resident_fixed_cycle import (
    FixedCyclePlan,
    load_fixed_cycle_registry,
)
from edge_runtime.resident_fixed_cycle_runtime import ResidentFixedCycleRuntime
from edge_runtime.remote import EdgeBehaviorExecutor


def _deployment(tmp_path: Path) -> tuple[FixedCyclePlan, object]:
    trajectories = {}
    for target_id in ("dig_01", "dig_02", "dig_03", "dump"):
        phase = "dump" if target_id == "dump" else "dig"
        template = {
            "schema_version": "resident_fixed_trajectory.v1",
            "trajectory_id": f"field-{target_id}-v1",
            "validation_status": "field_validated",
            "phase": phase,
            "frame_id": "machine_root_ros",
            "mission_id": "field_cycle_001",
            "mission_sha256": "c" * 64,
            "task_mode": "CarryMaterial" if phase == "dump" else "MoveToDig",
            "control_stage": "commissioning",
            "workspace_constraint": "field_validated",
            "waypoints": [[0.2, 0.0, 0.1], [1.0, 0.0, 0.0]],
            "waypoint_tolerance_m": 0.25,
            "intermediate_waypoint_tolerance_m": 0.40,
            "waypoint_dwell_s": 0.0,
            "tracking_timeout_s": 60.0,
        }
        payload = json.dumps(template, sort_keys=True).encode("utf-8")
        path = tmp_path / f"{target_id}.json"
        path.write_bytes(payload)
        trajectories[target_id] = {
            "trajectory_id": template["trajectory_id"],
            "phase": phase,
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    plan = FixedCyclePlan.from_mapping(
        {
            "schema_version": "resident_fixed_cycle_plan.v1",
            "plan_id": "field-cycle-v3a",
            "validation_status": "field_validated",
            "dig_sequence": ["dig_01", "dig_02", "dig_03"],
            "act_max_steps": 130,
            "trajectories": trajectories,
        }
    )
    return plan, load_fixed_cycle_registry(plan)


@dataclass
class _ActSegment:
    generation: int | None = None
    max_steps: int | None = None
    completed_steps: int = 0
    complete: bool = False


class _Core:
    def __init__(self) -> None:
        self.rl_is_active = True
        self.act_is_active = False
        self.is_operational = True
        self.segment = _ActSegment()
        self.calls = []
        self._generation = 10

    def activate_rl(self) -> int:
        self.calls.append(("activate_rl",))
        return self._generation

    def renew_mission_lease(self, *, lease_ms: int) -> None:
        self.calls.append(("renew_mission_lease", lease_ms))

    def activate_act(self, *, max_steps: int) -> int:
        self._generation += 1
        self.calls.append(("activate_act", max_steps))
        self.rl_is_active = False
        self.act_is_active = True
        self.segment = _ActSegment(self._generation, max_steps)
        return self._generation

    def control_status_snapshot(self):
        return SimpleNamespace(
            act_segment=self.segment,
            rl_is_active=self.rl_is_active,
            act_is_active=self.act_is_active,
            is_operational=self.is_operational,
        )

    def terminal_disarm(self):
        self.calls.append(("terminal_disarm",))
        self.rl_is_active = False
        self.act_is_active = False

    def complete_act(self) -> None:
        self.act_is_active = False
        self.rl_is_active = True
        self.segment = _ActSegment(
            self.segment.generation,
            self.segment.max_steps,
            self.segment.max_steps or 0,
            True,
        )


class _BehaviorExecutor:
    def __init__(self) -> None:
        self.busy = False
        self.requests = []
        self._sink = None

    def run_when_idle(self, operation):
        if self.busy:
            raise RuntimeError("an RL behavior is active")
        return operation()

    def handle(self, request, event_sink) -> None:
        if self.busy:
            raise RuntimeError("unexpected concurrent behavior")
        self.busy = True
        self.requests.append(request)
        self._sink = event_sink
        event_sink(
            {
                "type": "accepted",
                "request_id": request["request_id"],
            }
        )

    def succeed(self, *, reason_code: str) -> None:
        request = self.requests[-1]
        sink = self._sink
        self.busy = False
        sink(
            {
                "type": "result",
                "request_id": request["request_id"],
                "outcome": "SUCCEEDED",
                "reason_code": reason_code,
                "quiescence_confirmed": True,
            }
        )

    def feedback(self) -> None:
        self._sink(
            {
                "type": "feedback",
                "trajectory_waypoints": [
                    [0.8, 0.2, -0.1],
                    [0.9, 0.1, -0.05],
                    [1.0, 0.0, 0.0],
                ],
                "waypoint_index": 1,
                "waypoint_tolerance_m": 0.40,
            }
        )

    def fixed_action_feedback(self) -> None:
        self._sink(
            {
                "type": "feedback",
                "behavior": "ExecuteDump",
                "step_index": 0,
                "step_label": "open_bucket",
                "phase": "ACTIVE",
                "max_error": 0.01,
            }
        )

    def malformed_follow_feedback(self) -> None:
        self._sink(
            {
                "type": "feedback",
                "trajectory_waypoints": [],
                "waypoint_index": 0,
                "waypoint_tolerance_m": 0.25,
            }
        )


def test_follow_feedback_exposes_read_only_active_trajectory(tmp_path: Path) -> None:
    plan, registry = _deployment(tmp_path)
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=_Core(),
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )

    runtime.start(run_id="run-v3a-preview", requested_cycles=1)
    assert runtime.visualization_snapshot is None

    behaviors.feedback()

    assert runtime.visualization_snapshot == {
        "frame_id": "machine_root_ros",
        "target_id": "dig_01",
        "waypoints": (
            (0.8, 0.2, -0.1),
            (0.9, 0.1, -0.05),
            (1.0, 0.0, 0.0),
        ),
        "current_waypoint_index": 1,
        "waypoint_tolerance_m": 0.40,
    }

    behaviors.succeed(reason_code="SUCCEEDED")
    assert runtime.visualization_snapshot is None


def test_two_cycles_advance_on_orin_without_external_stage_commands(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )

    runtime.start(run_id="run-v3a-001", requested_cycles=2)
    assert behaviors.requests[-1]["type"] == "start_follow"
    assert behaviors.requests[-1]["trajectory"]["mission_phase"] == "dig"
    assert behaviors.requests[-1]["trajectory"]["waypoint_tolerance_m"] == 0.25
    assert (
        behaviors.requests[-1]["trajectory"]["intermediate_waypoint_tolerance_m"]
        == 0.40
    )

    behaviors.succeed(reason_code="SUCCEEDED")
    assert core.calls[-1] == ("activate_act", 130)

    core.complete_act()
    runtime.tick()
    assert behaviors.requests[-1]["trajectory"]["mission_phase"] == "dump"
    behaviors.succeed(reason_code="SUCCEEDED")
    assert behaviors.requests[-1]["type"] == "start_fixed_action"
    behaviors.succeed(reason_code="SEQUENCE_COMPLETED")
    assert behaviors.requests[-1]["trajectory"]["trajectory_id"] == "field-dig_02-v1"

    behaviors.succeed(reason_code="SUCCEEDED")
    core.complete_act()
    runtime.tick()
    behaviors.succeed(reason_code="SUCCEEDED")
    behaviors.succeed(reason_code="SEQUENCE_COMPLETED")
    assert behaviors.requests[-1]["trajectory"]["trajectory_id"] == "field-dig_02-v1"
    assert runtime.snapshot.stage == "FOLLOW_DIG"
    assert runtime.snapshot.completed_cycles == 2

    behaviors.succeed(reason_code="SUCCEEDED")

    assert runtime.snapshot.stage == "COMPLETED"
    assert runtime.snapshot.completed_cycles == 2
    assert core.calls[-1] == ("terminal_disarm",)
    assert [call for call in core.calls if call[0] == "activate_act"] == [
        ("activate_act", 130),
        ("activate_act", 130),
    ]


def test_fixed_action_feedback_does_not_require_follow_visualization(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )

    runtime.start(run_id="run-v3a-fixed-action-feedback", requested_cycles=1)
    behaviors.succeed(reason_code="SUCCEEDED")
    core.complete_act()
    runtime.tick()
    behaviors.succeed(reason_code="SUCCEEDED")
    assert behaviors.requests[-1]["type"] == "start_fixed_action"

    behaviors.fixed_action_feedback()

    assert runtime.snapshot.stage == "EXECUTE_DUMP"
    assert runtime.visualization_snapshot is None
    behaviors.succeed(reason_code="SEQUENCE_COMPLETED")
    assert behaviors.requests[-1]["type"] == "start_follow"


def test_malformed_read_only_visualization_does_not_abort_follow(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=_Core(),
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )
    runtime.start(run_id="run-v3a-bad-preview", requested_cycles=1)

    behaviors.malformed_follow_feedback()

    assert runtime.snapshot.stage == "FOLLOW_DIG"
    assert runtime.visualization_snapshot is None


def test_act_to_rl_dispatch_waits_for_acknowledged_rl_authority(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )
    runtime.start(run_id="run-v3a-ack", requested_cycles=1)
    behaviors.succeed(reason_code="SUCCEEDED")

    core.segment = _ActSegment(11, 130, 130, True)
    core.act_is_active = False
    core.rl_is_active = False
    runtime.tick()
    assert len(behaviors.requests) == 1

    core.rl_is_active = True
    runtime.tick()
    assert len(behaviors.requests) == 2
    assert behaviors.requests[-1]["trajectory"]["mission_phase"] == "dump"


def test_worker_loss_and_cancel_fail_closed_with_one_terminal_disarm(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    ready = [True]
    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: ready[0],
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )
    runtime.start(run_id="run-v3a-loss", requested_cycles=1)
    behaviors.succeed(reason_code="SUCCEEDED")
    ready[0] = False

    runtime.tick()

    assert runtime.snapshot.stage == "FAILED"
    assert runtime.snapshot.reason_code == "ACT_WORKER_UNAVAILABLE"
    assert core.calls.count(("terminal_disarm",)) == 1
    with pytest.raises(RuntimeError, match="no active"):
        runtime.cancel()


def test_terminal_status_remains_readable_before_owner_release(tmp_path: Path) -> None:
    plan, registry = _deployment(tmp_path)
    now = [10.0]
    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: now[0],
        terminal_status_grace_s=3.0,
    )
    runtime.start(run_id="run-v3a-receipt", requested_cycles=1)

    runtime.cancel()

    assert runtime.snapshot.stage == "CANCELLED"
    assert runtime.owner_release_ready is False
    now[0] = 12.99
    assert runtime.owner_release_ready is False
    now[0] = 13.0
    assert runtime.owner_release_ready is True


def test_only_pc_heartbeat_renews_active_fixed_cycle_lease(tmp_path: Path) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=_BehaviorExecutor(),
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )
    runtime.start(run_id="run-v3a-heartbeat", requested_cycles=1)
    initial = list(core.calls)

    runtime.tick()
    assert core.calls == initial

    runtime.heartbeat()
    assert core.calls[-1] == ("renew_mission_lease", 3000)


def test_heartbeat_returns_terminal_receipt_without_renewing_lease(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=_BehaviorExecutor(),
        act_worker_ready=lambda: True,
    )
    runtime.start(run_id="run-v3a-terminal-receipt", requested_cycles=1)
    runtime.cancel()
    calls_before_receipt = list(core.calls)

    receipt = runtime.heartbeat()

    assert receipt.stage == "CANCELLED"
    assert receipt.terminal is True
    assert core.calls == calls_before_receipt


def test_materialized_fixed_template_passes_the_real_behavior_boundary(
    tmp_path: Path,
) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    created = []
    executor = EdgeBehaviorExecutor(
        runtime_factory=SimpleNamespace(
            create=lambda snapshot: created.append(snapshot) or object(),
            trajectory_controller_backend="onnx_rl",
        ),
        runner_factory=lambda _runtime: SimpleNamespace(
            action_datagrams=0,
            close=lambda **_kwargs: None,
        ),
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
        sender_constructed=True,
        resident_rl_authorized=lambda: core.rl_is_active,
    )
    executor.observe(
        {
            "safety": {
                "control_enabled": True,
                "sensor_valid": True,
                "stm32_alive": True,
                "estop": False,
                "fault_flags": [],
            }
        }
    )
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=executor,
        act_worker_ready=lambda: True,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 10.0,
    )

    runtime.start(run_id="run-real-boundary", requested_cycles=1)

    assert executor.busy
    assert created[0].trajectory_id == "field-dig_01-v1"
    assert created[0].computed_sha256() == created[0].trajectory_sha256


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"act_worker_ready": None}, "callable"),
        ({"activation_timeout_s": 0.0}, "activation_timeout"),
        ({"mission_lease_ms": 499}, "mission_lease"),
        ({"terminal_status_grace_s": 0.5}, "terminal_status_grace"),
    ],
)
def test_runtime_configuration_is_strict(tmp_path: Path, kwargs, message) -> None:
    plan, registry = _deployment(tmp_path)
    arguments = {
        "plan": plan,
        "registry": registry,
        "core": _Core(),
        "behavior_executor": _BehaviorExecutor(),
        "act_worker_ready": lambda: True,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ResidentFixedCycleRuntime(**arguments)


def test_idle_and_nonoperational_runtime_boundaries_fail_closed(tmp_path: Path) -> None:
    plan, registry = _deployment(tmp_path)
    core = _Core()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=_BehaviorExecutor(),
        act_worker_ready=lambda: True,
    )

    assert runtime.tick().stage == "IDLE"
    with pytest.raises(RuntimeError, match="no active"):
        runtime.heartbeat()

    runtime.start(run_id="run-v3a-core-loss", requested_cycles=1)
    core.is_operational = False
    runtime.tick()
    assert runtime.snapshot.reason_code == "RESIDENT_CORE_NOT_OPERATIONAL"
    assert core.calls[-1] == ("terminal_disarm",)


def test_act_activation_and_authority_failures_are_terminal(tmp_path: Path) -> None:
    plan, registry = _deployment(tmp_path)

    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: False,
    )
    runtime.start(run_id="run-v3a-not-ready", requested_cycles=1)
    with pytest.raises(RuntimeError, match="local dispatch failed"):
        behaviors.succeed(reason_code="SUCCEEDED")
    assert runtime.snapshot.reason_code == "LOCAL_DISPATCH_FAILED"

    core = _Core()
    behaviors = _BehaviorExecutor()
    runtime = ResidentFixedCycleRuntime(
        plan=plan,
        registry=registry,
        core=core,
        behavior_executor=behaviors,
        act_worker_ready=lambda: True,
    )
    runtime.start(run_id="run-v3a-authority-loss", requested_cycles=1)
    behaviors.succeed(reason_code="SUCCEEDED")
    core.act_is_active = False
    runtime.tick()
    assert runtime.snapshot.reason_code == "ACT_AUTHORITY_LOST"
