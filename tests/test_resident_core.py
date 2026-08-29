import json
import unittest

from edge_runtime.resident_core import (
    AxisManualActionDeadzone,
    ManualActionDeadzoneContract,
    ResidentMotionCore,
)
from edge_runtime.resident_motion import ControlMode, MotionCandidate, ZERO_ACTION
from edge_runtime.resident_protocol import encode_motion_candidate
from edge_runtime.resident_sink import ResidentTelemetry


class RecordingSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None


class RecordingActionAudit:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event_type: str, **fields: object) -> bool:
        self.events.append({"event_type": event_type, **fields})
        return True


def uniform_deadzone_contract(
    value: float,
) -> ManualActionDeadzoneContract:
    axis = AxisManualActionDeadzone(value, value)
    return ManualActionDeadzoneContract((axis, axis, axis, axis))


def telemetry(
    *,
    receive_ns: int,
    command_seq: int,
    valid: bool,
    mode: ControlMode | None,
    action=ZERO_ACTION,
) -> ResidentTelemetry:
    return ResidentTelemetry(
        receive_monotonic_ns=receive_ns,
        command_rx_seq=command_seq,
        command_valid=valid,
        command_timed_out=False,
        control_mode=mode,
        command_action=action,
        control_enabled=True,
        estop=False,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
    )


class ResidentMotionCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = RecordingSerial()
        self.core = ResidentMotionCore(
            self.serial,
            max_state_age_ms=200.0,
            manual_action_deadzone_contract=uniform_deadzone_contract(0.15),
            wall_time_ms=lambda: 10_000,
            monotonic_ns=lambda: 1_090_000_000,
        )
        self.assertFalse(self.core.is_operational)
        self.assertFalse(self.core.rl_is_active)
        self.assertFalse(self.core.act_is_active)
        self.core.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        self.assertTrue(self.core.is_operational)
        self.assertFalse(self.core.rl_is_active)

    def packets(self) -> list[dict]:
        return [json.loads(payload.decode("ascii")) for payload in self.serial.writes]

    def acknowledge_latest_zero(self, *, mode: ControlMode, receive_ns: int) -> None:
        latest = self.packets()[-1]
        self.core.observe_telemetry(
            telemetry(
                receive_ns=receive_ns,
                command_seq=latest["command_seq"],
                valid=True,
                mode=mode,
            )
        )

    def test_rl_to_act_handoff_uses_one_owner_and_two_acknowledged_zeroes(self) -> None:
        rl_generation = self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )
        self.assertEqual(self.core.snapshot().generation, rl_generation)
        self.assertTrue(self.core.rl_is_active)
        self.assertFalse(self.core.act_is_active)
        self.assertIsNone(self.core.active_act_generation)

        act_generation = self.core.activate_act(now_monotonic_ns=1_040_000_000)
        self.assertFalse(self.core.rl_is_active)
        self.assertFalse(self.core.act_is_active)
        terminal_rl_zero = self.packets()[-1]
        self.assertEqual(
            terminal_rl_zero["schema_version"], "stm32_velocity_command.v1"
        )
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_060_000_000,
        )
        target_act_zero = self.packets()[-1]
        self.assertEqual(target_act_zero["schema_version"], "stm32_manual_command.v1")
        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_080_000_000,
        )
        self.assertFalse(self.core.rl_is_active)
        self.assertTrue(self.core.act_is_active)
        self.assertEqual(self.core.active_act_generation, act_generation)

        result = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=act_generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.1, -0.2, 0.3, 0.0),
                    created_monotonic_ns=1_025_000_000,
                    valid_until_monotonic_ns=1_100_000_000,
                )
            )
        )

        self.assertTrue(result.accepted)
        packet = self.packets()[-1]
        self.assertEqual((packet["Y2"], packet["Y1"], packet["X2"], packet["X1"]), (0.1, -0.2, 0.3, 0.0))
        self.assertIsNone(self.core.snapshot().last_handoff_latency_ms)
        self.core.observe_telemetry(
            telemetry(
                receive_ns=1_090_000_000,
                command_seq=result.command_seq,
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
                action=result.effective_action,
            )
        )
        self.assertEqual(self.core.snapshot().last_handoff_latency_ms, 30.0)

    def test_terminal_disarm_is_final_for_both_policy_adapters(self) -> None:
        self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )

        final = self.core.terminal_disarm(now_monotonic_ns=1_040_000_000)

        self.assertTrue(final.write_performed)
        self.assertEqual(final.effective_action, ZERO_ACTION)
        self.assertFalse(self.core.is_operational)
        with self.assertRaisesRegex(RuntimeError, "disarmed"):
            self.core.activate_act(now_monotonic_ns=1_050_000_000)

    def test_expired_mission_lease_terminally_disarms_the_owner(self) -> None:
        self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )
        self.core.renew_mission_lease(
            lease_ms=1500,
            now_monotonic_ns=1_030_000_000,
        )
        self.assertTrue(self.core.mission_lease_is_active)
        self.core.observe_telemetry(
            telemetry(
                receive_ns=2_400_000_000,
                command_seq=self.packets()[-1]["command_seq"],
                valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )

        writes_before_expiry = len(self.serial.writes)
        keepalive = self.core.tick(now_monotonic_ns=2_529_999_999)
        self.assertIsNotNone(keepalive)
        assert keepalive is not None
        self.assertEqual(keepalive.reason, "active_policy_idle_keepalive")
        self.assertEqual(keepalive.effective_action, ZERO_ACTION)
        self.assertEqual(len(self.serial.writes), writes_before_expiry + 1)

        expired = self.core.tick(now_monotonic_ns=2_530_000_000)

        self.assertIsNotNone(expired)
        self.assertTrue(expired.write_performed)
        self.assertEqual(expired.effective_action, ZERO_ACTION)
        self.assertFalse(self.core.mission_lease_is_active)
        self.assertFalse(self.core.is_operational)
        with self.assertRaisesRegex(RuntimeError, "disarmed"):
            self.core.renew_mission_lease(
                lease_ms=1500,
                now_monotonic_ns=2_540_000_000,
            )

    def test_expired_mission_lease_cannot_be_revived_before_tick(self) -> None:
        self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )
        self.core.renew_mission_lease(
            lease_ms=1500,
            now_monotonic_ns=1_030_000_000,
        )

        with self.assertRaisesRegex(RuntimeError, "expired"):
            self.core.renew_mission_lease(
                lease_ms=1500,
                now_monotonic_ns=2_530_000_000,
            )

        self.assertFalse(self.core.mission_lease_is_active)
        self.assertFalse(self.core.is_operational)
        self.assertEqual(self.packets()[-1]["boom_mps"], 0.0)

    def test_unarmed_mission_lease_does_not_expire_during_owner_startup(self) -> None:
        self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )
        writes_before_tick = len(self.serial.writes)

        result = self.core.tick(now_monotonic_ns=1_100_000_000)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reason, "active_policy_idle_keepalive")
        self.assertEqual(result.effective_action, ZERO_ACTION)
        self.assertEqual(len(self.serial.writes), writes_before_tick + 1)
        self.assertTrue(self.core.is_operational)
        self.assertFalse(self.core.mission_lease_is_active)

    def test_repeated_terminal_disarm_preserves_the_exact_zero_for_ack_wait(self) -> None:
        self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )

        first = self.core.terminal_disarm(now_monotonic_ns=1_040_000_000)
        writes_after_first = len(self.serial.writes)
        repeated = self.core.terminal_disarm(now_monotonic_ns=1_050_000_000)

        self.assertTrue(first.write_performed)
        self.assertEqual(repeated, first)
        self.assertEqual(len(self.serial.writes), writes_after_first)

    def test_terminal_zero_ack_is_remembered_after_core_is_disarmed(self) -> None:
        self.core.activate_rl(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_020_000_000,
        )
        terminal = self.core.terminal_disarm(now_monotonic_ns=1_040_000_000)

        self.assertFalse(self.core.terminal_zero_acknowledged)
        self.core.observe_telemetry(
            telemetry(
                receive_ns=1_060_000_000,
                command_seq=terminal.command_seq,
                valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )

        self.assertTrue(self.core.terminal_zero_acknowledged)

    def test_act_candidate_lease_expiry_zeros_and_revokes_the_generation(self) -> None:
        generation = self.core.activate_act(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_020_000_000,
        )
        moving = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.4, 0.0, 0.0, 0.0),
                    created_monotonic_ns=1_025_000_000,
                    valid_until_monotonic_ns=1_100_000_000,
                )
            )
        )
        self.assertTrue(moving.accepted)

        writes_while_lease_is_valid = len(self.serial.writes)
        self.core.tick(now_monotonic_ns=1_100_000_000)
        self.assertEqual(len(self.serial.writes), writes_while_lease_is_valid)

        self.core.tick(now_monotonic_ns=1_101_000_000)

        self.assertEqual(len(self.serial.writes), writes_while_lease_is_valid + 1)
        self.assertEqual(self.packets()[-1]["Y2"], 0.0)
        self.assertFalse(self.core.act_is_active)
        self.assertIsNone(self.core.active_act_generation)
        replacement_generation = self.core.activate_act(
            now_monotonic_ns=1_102_000_000
        )
        self.assertNotEqual(replacement_generation, generation)

    def test_act_worker_disconnect_immediately_zeros_and_revokes_motion(self) -> None:
        generation = self.core.activate_act(now_monotonic_ns=1_000_000_000)
        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_020_000_000,
        )
        moving = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.0, -0.4, 0.2, 0.0),
                    created_monotonic_ns=1_025_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertTrue(moving.accepted)
        writes_before_disconnect = len(self.serial.writes)

        revoked = self.core.notify_act_worker_disconnected(
            now_monotonic_ns=1_040_000_000
        )

        self.assertTrue(revoked)
        self.assertEqual(len(self.serial.writes), writes_before_disconnect + 1)
        self.assertEqual(
            tuple(
                self.packets()[-1][name]
                for name in ("Y2", "Y1", "X2", "X1")
            ),
            ZERO_ACTION,
        )
        self.assertFalse(self.core.act_is_active)
        replacement_generation = self.core.activate_act(
            now_monotonic_ns=1_050_000_000
        )
        self.assertNotEqual(replacement_generation, generation)

    def test_expected_act_disconnect_after_terminal_disarm_is_a_noop(self) -> None:
        self.core.activate_act(
            max_steps=130,
            now_monotonic_ns=1_000_000_000,
        )
        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_020_000_000,
        )
        self.core.terminal_disarm(now_monotonic_ns=1_030_000_000)
        writes_after_disarm = len(self.serial.writes)

        revoked = self.core.notify_act_worker_disconnected(
            now_monotonic_ns=1_040_000_000
        )

        self.assertFalse(revoked)
        self.assertEqual(len(self.serial.writes), writes_after_disarm)

    def test_bounded_act_segment_waits_for_last_action_ack_then_returns_to_rl(self) -> None:
        generation = self.core.activate_act(
            max_steps=2,
            now_monotonic_ns=1_000_000_000,
        )
        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_020_000_000,
        )

        first = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.1, 0.0, 0.0, 0.0),
                    created_monotonic_ns=1_025_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        first_status = self.core.act_segment_snapshot()
        self.assertTrue(first.accepted)
        self.assertEqual(first_status.generation, generation)
        self.assertEqual(first_status.max_steps, 2)
        self.assertEqual(first_status.completed_steps, 1)
        self.assertFalse(first_status.complete)

        second_action = (0.0, -0.23449, 0.34549, 0.0)
        second_action_echo = (0.0, -0.234, 0.345, 0.0)
        second = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=second_action,
                    created_monotonic_ns=1_030_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        waiting_status = self.core.act_segment_snapshot()
        self.assertTrue(second.accepted)
        self.assertEqual(waiting_status.completed_steps, 2)
        self.assertFalse(waiting_status.complete)
        self.assertTrue(self.core.act_is_active)

        writes_at_budget = len(self.serial.writes)
        over_budget = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.8, 0.8, 0.8, 0.0),
                    created_monotonic_ns=1_035_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertFalse(over_budget.write_performed)
        self.assertEqual(over_budget.reason, "act_segment_budget_reached")
        self.assertEqual(len(self.serial.writes), writes_at_budget)
        self.assertEqual(self.core.act_segment_snapshot().completed_steps, 2)

        self.core.observe_telemetry(
            telemetry(
                receive_ns=1_040_000_000,
                command_seq=(second.command_seq - 1) & 0xFFFFFFFF,
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
                action=second_action_echo,
            )
        )
        self.assertTrue(self.core.act_is_active)
        self.assertFalse(self.core.act_segment_snapshot().complete)

        self.core.observe_telemetry(
            telemetry(
                receive_ns=1_050_000_000,
                command_seq=second.command_seq,
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
                action=second_action_echo,
            )
        )
        completed_status = self.core.act_segment_snapshot()
        self.assertTrue(completed_status.complete)
        self.assertEqual(completed_status.completed_steps, 2)
        self.assertFalse(self.core.act_is_active)
        self.assertFalse(self.core.rl_is_active)
        self.assertEqual(
            self.packets()[-1]["schema_version"],
            "stm32_manual_command.v1",
        )

        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_070_000_000,
        )
        self.assertEqual(
            self.packets()[-1]["schema_version"],
            "stm32_velocity_command.v1",
        )
        self.acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_090_000_000,
        )
        self.assertTrue(self.core.rl_is_active)

    def test_act_segment_audit_exposes_the_deadzone_tail_at_the_step_budget(self) -> None:
        serial = RecordingSerial()
        audit = RecordingActionAudit()
        core = ResidentMotionCore(
            serial,
            max_state_age_ms=200.0,
            manual_action_deadzone_contract=uniform_deadzone_contract(0.15),
            wall_time_ms=lambda: 10_000,
            monotonic_ns=lambda: 1_090_000_000,
            action_audit=audit,
        )
        core.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        generation = core.activate_act(
            max_steps=2,
            now_monotonic_ns=1_000_000_000,
        )
        claim = json.loads(serial.writes[-1].decode("ascii"))
        core.observe_telemetry(
            telemetry(
                receive_ns=1_020_000_000,
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        for created_ns, action in (
            (1_025_000_000, (0.2, 0.0, 0.0, 0.0)),
            (1_030_000_000, (0.1, -0.15, 0.0, 0.0)),
        ):
            result = core.submit_act(
                encode_motion_candidate(
                    MotionCandidate(
                        source="act_dig",
                        generation=generation,
                        mode=ControlMode.MANUAL_ACTION,
                        action=action,
                        created_monotonic_ns=created_ns,
                        valid_until_monotonic_ns=1_200_000_000,
                    )
                )
            )
            self.assertTrue(result.accepted)

        steps = [
            event for event in audit.events if event["event_type"] == "act_step"
        ]
        summaries = [
            event
            for event in audit.events
            if event["event_type"] == "act_segment_summary"
        ]
        self.assertEqual(
            [(step["completed_steps"], step["all_axes_in_deadzone"]) for step in steps],
            [(1, False), (2, True)],
        )
        writes = [
            event
            for event in audit.events
            if event["event_type"] == "command_write"
        ]
        self.assertEqual(steps[0]["runtime_id"], writes[-1]["runtime_id"])
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["last_effective_step"], 1)
        self.assertEqual(summaries[0]["trailing_deadzone_steps"], 1)
        self.assertEqual(summaries[0]["estimated_trailing_deadzone_ms"], 100.0)

    def test_act_deadzone_uses_the_injected_directional_contract(self) -> None:
        serial = RecordingSerial()
        audit = RecordingActionAudit()
        core = ResidentMotionCore(
            serial,
            max_state_age_ms=200.0,
            wall_time_ms=lambda: 10_000,
            monotonic_ns=lambda: 1_090_000_000,
            action_audit=audit,
            manual_action_deadzone_contract=ManualActionDeadzoneContract(
                (
                    AxisManualActionDeadzone(0.12, 0.08),
                    AxisManualActionDeadzone(0.15, 0.15),
                    AxisManualActionDeadzone(0.15, 0.15),
                    AxisManualActionDeadzone(0.15, 0.15),
                )
            ),
        )
        core.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        generation = core.activate_act(
            max_steps=2,
            now_monotonic_ns=1_000_000_000,
        )
        claim = json.loads(serial.writes[-1].decode("ascii"))
        core.observe_telemetry(
            telemetry(
                receive_ns=1_020_000_000,
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        for created_ns, action in (
            (1_025_000_000, (0.1201, 0.0, 0.0, 0.0)),
            (1_030_000_000, (0.12, -0.08, 0.0, 0.0)),
        ):
            result = core.submit_act(
                encode_motion_candidate(
                    MotionCandidate(
                        source="act_dig",
                        generation=generation,
                        mode=ControlMode.MANUAL_ACTION,
                        action=action,
                        created_monotonic_ns=created_ns,
                        valid_until_monotonic_ns=1_200_000_000,
                    )
                )
            )
            self.assertTrue(result.accepted)

        steps = [
            event for event in audit.events if event["event_type"] == "act_step"
        ]
        self.assertEqual(
            [step["all_axes_in_deadzone"] for step in steps],
            [False, True],
        )
        self.assertEqual(
            [step["manual_deadzone_contract_enabled"] for step in steps],
            [True, True],
        )

    def test_act_early_completion_inspects_a_full_new_chunk_before_execution(self) -> None:
        serial = RecordingSerial()
        audit = RecordingActionAudit()
        core = ResidentMotionCore(
            serial,
            max_state_age_ms=200.0,
            wall_time_ms=lambda: 10_000,
            monotonic_ns=lambda: 1_090_000_000,
            action_audit=audit,
            manual_action_deadzone_contract=uniform_deadzone_contract(0.15),
            act_early_completion_chunk_steps=10,
            act_early_completion_min_steps=10,
        )
        core.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        generation = core.activate_act(
            max_steps=30,
            now_monotonic_ns=1_000_000_000,
        )
        claim = json.loads(serial.writes[-1].decode("ascii"))
        core.observe_telemetry(
            telemetry(
                receive_ns=1_020_000_000,
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )

        results = []
        for index in range(10):
            pre_minimum_chunk = ((0.1, 0.0, 0.0, 0.0),) * 10
            result = core.submit_act(
                encode_motion_candidate(
                    MotionCandidate(
                        source="act_dig",
                        generation=generation,
                        mode=ControlMode.MANUAL_ACTION,
                        action=(
                            (0.1, 0.0, 0.0, 0.0)
                            if index == 0
                            else (0.2, 0.0, 0.0, 0.0)
                        ),
                        action_chunk=pre_minimum_chunk if index == 0 else None,
                        created_monotonic_ns=1_025_000_000 + index,
                        valid_until_monotonic_ns=1_200_000_000,
                    )
                )
            )
            self.assertTrue(result.accepted)
            results.append(result)

        mixed_chunk = (
            *((0.1, 0.0, 0.0, 0.0),) * 9,
            (0.1501, 0.0, 0.0, 0.0),
        )
        mixed = core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=mixed_chunk[0],
                    action_chunk=mixed_chunk,
                    created_monotonic_ns=1_030_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertTrue(mixed.accepted)
        self.assertEqual(core.act_segment_snapshot().completed_steps, 11)

        deadzone_chunk = ((0.15, -0.15, 0.0, 0.0),) * 10
        final = core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=deadzone_chunk[0],
                    action_chunk=deadzone_chunk,
                    created_monotonic_ns=1_031_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertTrue(final.accepted)
        self.assertEqual(core.act_segment_snapshot().completed_steps, 12)
        self.assertFalse(core.act_segment_snapshot().complete)
        rejected = core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.8, 0.0, 0.0, 0.0),
                    created_monotonic_ns=1_030_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "act_segment_early_complete")

        summaries = [
            event
            for event in audit.events
            if event["event_type"] == "act_segment_summary"
        ]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["completion_reason"], "deadzone_chunk")
        self.assertEqual(summaries[0]["deadzone_chunk_steps"], 10)
        self.assertEqual(summaries[0]["skipped_budget_steps"], 18)

        core.observe_telemetry(
            telemetry(
                receive_ns=1_040_000_000,
                command_seq=final.command_seq,
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
                action=final.effective_action,
            )
        )
        self.assertTrue(core.act_segment_snapshot().complete)
        self.assertEqual(core.act_segment_snapshot().completed_steps, 12)

    def test_missing_deadzone_contract_disables_early_completion(self) -> None:
        serial = RecordingSerial()
        core = ResidentMotionCore(
            serial,
            max_state_age_ms=200.0,
            wall_time_ms=lambda: 10_000,
            monotonic_ns=lambda: 1_090_000_000,
            manual_action_deadzone_contract=None,
            act_early_completion_chunk_steps=10,
            act_early_completion_min_steps=10,
        )
        core.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        generation = core.activate_act(
            max_steps=30,
            now_monotonic_ns=1_000_000_000,
        )
        claim = json.loads(serial.writes[-1].decode("ascii"))
        core.observe_telemetry(
            telemetry(
                receive_ns=1_020_000_000,
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        deadzone_chunk = ((0.0, 0.0, 0.0, 0.0),) * 10
        final = core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=deadzone_chunk[0],
                    action_chunk=deadzone_chunk,
                    created_monotonic_ns=1_031_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertTrue(final.accepted)
        self.assertEqual(core.act_segment_snapshot().completed_steps, 1)
        self.assertFalse(core.act_segment_snapshot().complete)

        still_running = core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.8, 0.0, 0.0, 0.0),
                    created_monotonic_ns=1_032_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertTrue(still_running.accepted)

    def test_default_early_completion_starts_at_step_101_chunk_boundary(self) -> None:
        generation = self.core.activate_act(
            max_steps=130,
            now_monotonic_ns=1_000_000_000,
        )
        self.acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_020_000_000,
        )
        active_chunk = ((0.2, 0.0, 0.0, 0.0),) * 10
        for step_index in range(100):
            result = self.core.submit_act(
                encode_motion_candidate(
                    MotionCandidate(
                        source="act_dig",
                        generation=generation,
                        mode=ControlMode.MANUAL_ACTION,
                        action=active_chunk[step_index % 10],
                        action_chunk=(
                            active_chunk if step_index % 10 == 0 else None
                        ),
                        created_monotonic_ns=1_025_000_000 + step_index,
                        valid_until_monotonic_ns=1_200_000_000,
                    )
                )
            )
            self.assertTrue(result.accepted)
        self.assertEqual(self.core.act_segment_snapshot().completed_steps, 100)
        self.assertFalse(self.core.act_segment_snapshot().complete)

        deadzone_chunk = ((0.15, -0.15, 0.0, 0.0),) * 10
        final = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=deadzone_chunk[0],
                    action_chunk=deadzone_chunk,
                    created_monotonic_ns=1_026_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertTrue(final.accepted)
        self.assertEqual(self.core.act_segment_snapshot().completed_steps, 101)
        self.assertFalse(self.core.act_segment_snapshot().complete)
        rejected = self.core.submit_act(
            encode_motion_candidate(
                MotionCandidate(
                    source="act_dig",
                    generation=generation,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.8, 0.0, 0.0, 0.0),
                    created_monotonic_ns=1_027_000_000,
                    valid_until_monotonic_ns=1_200_000_000,
                )
            )
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "act_segment_early_complete")

    def test_repeated_act_activation_is_idempotent_but_cannot_change_budget(self) -> None:
        generation = self.core.activate_act(
            max_steps=130,
            now_monotonic_ns=1_000_000_000,
        )
        self.assertEqual(
            self.core.activate_act(
                max_steps=130,
                now_monotonic_ns=1_010_000_000,
            ),
            generation,
        )
        self.assertEqual(len(self.serial.writes), 1)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.core.activate_act(
                max_steps=131,
                now_monotonic_ns=1_010_000_000,
            )

    def test_act_step_budget_validation_is_fail_closed(self) -> None:
        for value in (True, 0, -1, 2001, 1.5, "130"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "max_steps"):
                    self.core.activate_act(max_steps=value)


if __name__ == "__main__":
    unittest.main()
