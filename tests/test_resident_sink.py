import json
import threading
import unittest

from edge_runtime.resident_motion import (
    ControlMode,
    MotionCandidate,
    PolicyBinding,
    ZERO_ACTION,
)
from edge_runtime.resident_sink import ResidentCommandSink, ResidentTelemetry


class RecordingSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.flush_count = 0
        self.write_started: threading.Event | None = None
        self.release_write: threading.Event | None = None
        self.short_write = False

    def write(self, payload: bytes) -> int:
        if self.write_started is not None:
            self.write_started.set()
        if self.release_write is not None:
            self.release_write.wait(timeout=1.0)
        self.writes.append(payload)
        return len(payload) - 1 if self.short_write else len(payload)

    def flush(self) -> None:
        self.flush_count += 1


def telemetry(
    *,
    receive_ns: int,
    command_rx_seq: int = 0,
    command_valid: bool = False,
    command_timed_out: bool = False,
    mode: ControlMode | None = None,
    action=ZERO_ACTION,
    control_enabled: bool = True,
    estop: bool = False,
    sensor_valid: bool = True,
    stm32_alive: bool = True,
    fault_flags: int = 0,
) -> ResidentTelemetry:
    return ResidentTelemetry(
        receive_monotonic_ns=receive_ns,
        command_rx_seq=command_rx_seq,
        command_valid=command_valid,
        command_timed_out=command_timed_out,
        control_mode=mode,
        command_action=action,
        control_enabled=control_enabled,
        estop=estop,
        sensor_valid=sensor_valid,
        stm32_alive=stm32_alive,
        fault_flags=fault_flags,
    )


def candidate(
    *,
    source: str,
    generation: int,
    mode: ControlMode,
    action=(0.1, -0.2, 0.3, -0.4),
    created_ns: int = 1_000_000_000,
    valid_until_ns: int = 1_200_000_000,
) -> MotionCandidate:
    return MotionCandidate(
        source=source,
        generation=generation,
        mode=mode,
        action=action,
        created_monotonic_ns=created_ns,
        valid_until_monotonic_ns=valid_until_ns,
    )


class ResidentCommandSinkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = RecordingSerial()
        self.sink = ResidentCommandSink(
            self.serial,
            max_state_age_ms=200.0,
            runtime_id="runtime-test-001",
        )
        self.sink.initialize(
            telemetry(receive_ns=990_000_000),
        )
        self.rl = PolicyBinding("rl_follow", ControlMode.VELOCITY_REFERENCE)
        self.act = PolicyBinding("act_dig", ControlMode.MANUAL_ACTION)

    def packets(self) -> list[dict]:
        return [json.loads(payload.decode("ascii")) for payload in self.serial.writes]

    def acknowledge_latest_zero(
        self,
        *,
        receive_ns: int,
        mode: ControlMode,
    ) -> None:
        packet = self.packets()[-1]
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=receive_ns,
                command_rx_seq=packet["command_seq"],
                command_valid=True,
                mode=mode,
                action=ZERO_ACTION,
            )
        )

    def activate(
        self,
        binding: PolicyBinding,
        *,
        now_ns: int = 1_000_000_000,
    ) -> int:
        generation = self.sink.request_handoff(binding, now_monotonic_ns=now_ns)
        self.acknowledge_latest_zero(
            receive_ns=now_ns + 20_000_000,
            mode=binding.mode,
        )
        return generation

    def test_first_activation_writes_target_schema_zero_then_allows_motion(self) -> None:
        generation = self.sink.request_handoff(
            self.act,
            now_monotonic_ns=1_000_000_000,
        )
        claim = self.packets()[-1]
        self.assertEqual(claim["schema_version"], "stm32_manual_command.v1")
        self.assertEqual(
            [claim[name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")],
            [0.0] * 6,
        )

        blocked = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
            ),
            now_monotonic_ns=1_010_000_000,
        )
        self.assertFalse(blocked.write_performed)
        self.assertEqual(blocked.reason, "handoff_in_progress")

        self.acknowledge_latest_zero(
            receive_ns=1_020_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )
        accepted = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
            ),
            now_monotonic_ns=1_030_000_000,
        )
        self.assertTrue(accepted.write_performed)
        self.assertTrue(accepted.accepted)
        self.assertEqual(self.packets()[-1]["X1"], -0.4)
        self.assertEqual(self.packets()[-1]["Y2"], 0.1)

    def test_active_policy_without_a_candidate_refreshes_the_safe_zero(self) -> None:
        generation = self.activate(self.rl)
        writes_after_activation = len(self.serial.writes)

        before_interval = self.sink.tick(now_monotonic_ns=1_090_000_000)
        self.assertIsNone(before_interval)
        self.assertEqual(len(self.serial.writes), writes_after_activation)

        keepalive = self.sink.tick(now_monotonic_ns=1_100_000_000)
        self.assertIsNotNone(keepalive)
        assert keepalive is not None
        self.assertTrue(keepalive.write_performed)
        self.assertFalse(keepalive.accepted)
        self.assertEqual(keepalive.reason, "active_policy_idle_keepalive")
        self.assertEqual(keepalive.effective_action, ZERO_ACTION)
        self.assertEqual(
            self.packets()[-1]["schema_version"],
            "stm32_velocity_command.v1",
        )
        self.assertEqual(self.sink.snapshot().generation, generation)
        self.assertEqual(self.sink.snapshot().active_binding, self.rl)

    def test_rl_to_act_uses_two_acknowledged_zeros_with_one_sequence(self) -> None:
        self.activate(self.rl)
        generation = self.sink.request_handoff(
            self.act,
            now_monotonic_ns=1_100_000_000,
        )

        terminal_velocity = self.packets()[-1]
        self.assertEqual(
            terminal_velocity["schema_version"],
            "stm32_velocity_command.v1",
        )
        self.acknowledge_latest_zero(
            receive_ns=1_120_000_000,
            mode=ControlMode.VELOCITY_REFERENCE,
        )

        target_manual = self.packets()[-1]
        self.assertEqual(target_manual["schema_version"], "stm32_manual_command.v1")
        self.assertEqual(
            target_manual["command_seq"],
            (terminal_velocity["command_seq"] + 1) & 0xFFFFFFFF,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_140_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )

        accepted = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                action=(0.12349, -0.23449, 0.34549, -0.45649),
                created_ns=1_140_000_000,
                valid_until_ns=1_240_000_000,
            ),
            now_monotonic_ns=1_150_000_000,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(
            self.packets()[-1]["command_seq"],
            (target_manual["command_seq"] + 1) & 0xFFFFFFFF,
        )
        self.assertIsNone(self.sink.snapshot().last_handoff_latency_ms)

        writes_before_second = len(self.serial.writes)
        waiting = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                action=(0.4, 0.0, 0.0, 0.0),
                created_ns=1_150_000_000,
                valid_until_ns=1_250_000_000,
            ),
            now_monotonic_ns=1_155_000_000,
        )
        self.assertFalse(waiting.accepted)
        self.assertFalse(waiting.write_performed)
        self.assertEqual(waiting.reason, "handoff_first_action_ack_pending")
        self.assertEqual(len(self.serial.writes), writes_before_second)

        zero_while_waiting = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                action=ZERO_ACTION,
                created_ns=1_155_000_000,
                valid_until_ns=1_255_000_000,
            ),
            now_monotonic_ns=1_160_000_000,
        )
        self.assertFalse(zero_while_waiting.accepted)
        self.assertFalse(zero_while_waiting.write_performed)
        self.assertEqual(
            zero_while_waiting.reason,
            "handoff_first_action_ack_pending",
        )
        self.assertEqual(len(self.serial.writes), writes_before_second)

        first_action = (0.123, -0.234, 0.345, -0.456)
        with self.assertLogs("edge_runtime.resident_handoff", level="INFO") as logs:
            self.sink.observe_telemetry(
                telemetry(
                    receive_ns=1_170_000_000,
                    command_rx_seq=accepted.command_seq,
                    command_valid=True,
                    mode=ControlMode.MANUAL_ACTION,
                    action=first_action,
                )
            )
        self.assertEqual(self.sink.snapshot().last_handoff_latency_ms, 50.0)
        metric_lines = [
            line.split("RESIDENT_HANDOFF_SAMPLE ", 1)[1]
            for line in logs.output
            if "RESIDENT_HANDOFF_SAMPLE " in line
        ]
        self.assertEqual(len(metric_lines), 1)
        self.assertEqual(
            json.loads(metric_lines[0]),
            {
                "first_command_ack_ms": 20.0,
                "first_nonzero_ack_monotonic_ns": 1_170_000_000,
                "first_nonzero_action": [0.12349, -0.23449, 0.34549, -0.45649],
                "first_nonzero_command_seq": accepted.command_seq,
                "first_nonzero_write_monotonic_ns": 1_150_000_000,
                "from_mode": "velocity_reference",
                "from_source": "rl_follow",
                "generation": generation,
                "latency_ms": 50.0,
                "policy_ready_wait_ms": 10.0,
                "runtime_id": "runtime-test-001",
                "schema_version": "resident_handoff_sample.v1",
                "target_zero_ack_monotonic_ns": 1_140_000_000,
                "target_zero_command_seq": target_manual["command_seq"],
                "terminal_zero_ack_monotonic_ns": 1_120_000_000,
                "terminal_zero_command_seq": terminal_velocity["command_seq"],
                "to_mode": "manual_action",
                "to_source": "act_dig",
                "zero_claim_ms": 20.0,
            },
        )

    def test_first_nonzero_ack_uses_sequence_and_mode_not_transformed_echo(self) -> None:
        self.activate(self.rl)
        generation = self.sink.request_handoff(
            self.act,
            now_monotonic_ns=1_100_000_000,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_120_000_000,
            mode=ControlMode.VELOCITY_REFERENCE,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_140_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )
        accepted = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_140_000_000,
                valid_until_ns=1_240_000_000,
            ),
            now_monotonic_ns=1_150_000_000,
        )

        with self.assertLogs("edge_runtime.resident_handoff", level="INFO"):
            self.sink.observe_telemetry(
                telemetry(
                    receive_ns=1_160_000_000,
                    command_rx_seq=accepted.command_seq,
                    command_valid=True,
                    mode=ControlMode.MANUAL_ACTION,
                    action=(0.4, 0.0, 0.0, 0.0),
                )
            )

        self.assertEqual(self.sink.snapshot().last_handoff_latency_ms, 40.0)
        next_action = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_160_000_000,
                valid_until_ns=1_260_000_000,
            ),
            now_monotonic_ns=1_170_000_000,
        )
        self.assertTrue(next_action.accepted)
        self.assertTrue(next_action.write_performed)

    def test_unsafe_telemetry_revokes_generation_while_first_action_ack_is_pending(self) -> None:
        self.activate(self.rl)
        generation = self.sink.request_handoff(
            self.act,
            now_monotonic_ns=1_100_000_000,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_120_000_000,
            mode=ControlMode.VELOCITY_REFERENCE,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_140_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )
        first = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_140_000_000,
                valid_until_ns=1_300_000_000,
            ),
            now_monotonic_ns=1_150_000_000,
        )
        self.assertTrue(first.accepted)

        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_160_000_000,
                command_rx_seq=(first.command_seq - 1) & 0xFFFFFFFF,
                command_valid=True,
                mode=ControlMode.MANUAL_ACTION,
                control_enabled=False,
            )
        )

        revoked = self.sink.snapshot()
        self.assertNotEqual(revoked.generation, generation)
        self.assertEqual(revoked.phase.value, "terminal_zero_pending")
        self.assertEqual(self.packets()[-1]["Y2"], 0.0)
        self.acknowledge_latest_zero(
            receive_ns=1_170_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )
        stale = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_170_000_000,
                valid_until_ns=1_300_000_000,
            ),
            now_monotonic_ns=1_180_000_000,
        )
        self.assertFalse(stale.accepted)
        self.assertFalse(stale.write_performed)

        replacement = self.sink.request_handoff(
            self.act,
            now_monotonic_ns=1_190_000_000,
        )
        self.assertNotEqual(replacement, generation)
        self.acknowledge_latest_zero(
            receive_ns=1_200_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )
        resumed = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=replacement,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_200_000_000,
                valid_until_ns=1_300_000_000,
            ),
            now_monotonic_ns=1_210_000_000,
        )
        self.assertTrue(resumed.accepted)

    def test_cold_activation_does_not_report_a_policy_handoff_latency(self) -> None:
        generation = self.activate(self.act)
        accepted = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_020_000_000,
                valid_until_ns=1_200_000_000,
            ),
            now_monotonic_ns=1_030_000_000,
        )
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_050_000_000,
                command_rx_seq=accepted.command_seq,
                command_valid=True,
                mode=ControlMode.MANUAL_ACTION,
                action=accepted.effective_action,
            )
        )

        self.assertIsNone(self.sink.snapshot().last_handoff_latency_ms)

    def test_old_policy_cannot_zero_the_new_policy_after_handoff_completes(self) -> None:
        rl_generation = self.activate(self.rl)
        act_generation = self.sink.request_handoff(
            self.act,
            now_monotonic_ns=1_100_000_000,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_120_000_000,
            mode=ControlMode.VELOCITY_REFERENCE,
        )
        self.acknowledge_latest_zero(
            receive_ns=1_140_000_000,
            mode=ControlMode.MANUAL_ACTION,
        )
        accepted = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=act_generation,
                mode=ControlMode.MANUAL_ACTION,
                created_ns=1_140_000_000,
                valid_until_ns=1_240_000_000,
            ),
            now_monotonic_ns=1_150_000_000,
        )
        self.assertTrue(accepted.accepted)
        writes_before = len(self.serial.writes)

        stale = self.sink.submit_candidate(
            candidate(
                source="rl_follow",
                generation=rl_generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                created_ns=1_140_000_000,
                valid_until_ns=1_240_000_000,
            ),
            now_monotonic_ns=1_160_000_000,
        )

        self.assertFalse(stale.accepted)
        self.assertFalse(stale.write_performed)
        self.assertEqual(stale.reason, "wrong_source")
        self.assertEqual(len(self.serial.writes), writes_before)

    def test_wrong_or_timed_out_ack_cannot_advance_the_handoff(self) -> None:
        self.sink.request_handoff(self.act, now_monotonic_ns=1_000_000_000)
        sequence = self.packets()[-1]["command_seq"]

        rejected = (
            telemetry(
                receive_ns=1_010_000_000,
                command_rx_seq=sequence,
                command_valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            ),
            telemetry(
                receive_ns=1_020_000_000,
                command_rx_seq=sequence,
                command_timed_out=True,
                mode=ControlMode.MANUAL_ACTION,
            ),
            telemetry(
                receive_ns=1_030_000_000,
                command_rx_seq=(sequence + 1) & 0xFFFFFFFF,
                command_valid=True,
                mode=ControlMode.MANUAL_ACTION,
            ),
        )
        for frame in rejected:
            self.sink.observe_telemetry(frame)
            self.assertEqual(self.sink.snapshot().phase.value, "target_zero_pending")

    def test_unsafe_or_stale_state_forces_zero_at_the_final_write_boundary(self) -> None:
        generation = self.activate(self.rl)
        unsafe = self.sink.submit_candidate(
            candidate(
                source="rl_follow",
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
            ),
            now_monotonic_ns=1_300_000_000,
        )
        self.assertFalse(unsafe.accepted)
        self.assertTrue(unsafe.write_performed)
        self.assertEqual(unsafe.reason, "safety_rejected")
        self.assertEqual(
            [self.packets()[-1][name] for name in ("boom_mps", "stick_mps", "bucket_mps", "swing_radps")],
            [0.0] * 4,
        )

    def test_nonzero_candidate_lease_expiry_zeros_and_revokes_the_source(self) -> None:
        generation = self.activate(self.rl)
        moving = self.sink.submit_candidate(
            candidate(
                source="rl_follow",
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                action=(0.2, 0.0, 0.0, 0.0),
                created_ns=1_025_000_000,
                valid_until_ns=1_100_000_000,
            ),
            now_monotonic_ns=1_030_000_000,
        )
        self.assertTrue(moving.accepted)
        writes_before_expiry = len(self.serial.writes)

        still_valid = self.sink.tick(now_monotonic_ns=1_100_000_000)
        self.assertIsNone(still_valid)
        expired = self.sink.tick(now_monotonic_ns=1_101_000_000)

        self.assertIsNotNone(expired)
        self.assertEqual(expired.reason, "candidate_lease_expired")
        self.assertEqual(expired.effective_action, ZERO_ACTION)
        self.assertEqual(len(self.serial.writes), writes_before_expiry + 1)
        self.assertEqual(self.packets()[-1]["boom_mps"], 0.0)
        stale = self.sink.submit_candidate(
            candidate(
                source="rl_follow",
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
                action=(0.3, 0.0, 0.0, 0.0),
                created_ns=1_101_000_000,
                valid_until_ns=1_200_000_000,
            ),
            now_monotonic_ns=1_102_000_000,
        )
        self.assertFalse(stale.accepted)
        self.assertFalse(stale.write_performed)

    def test_accepted_zero_clears_the_nonzero_candidate_lease(self) -> None:
        generation = self.activate(self.act)
        moving = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                action=(0.2, 0.0, 0.0, 0.0),
                created_ns=1_025_000_000,
                valid_until_ns=1_050_000_000,
            ),
            now_monotonic_ns=1_030_000_000,
        )
        stopped = self.sink.submit_candidate(
            candidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                action=ZERO_ACTION,
                created_ns=1_035_000_000,
                valid_until_ns=1_200_000_000,
            ),
            now_monotonic_ns=1_040_000_000,
        )
        self.assertTrue(moving.accepted)
        self.assertTrue(stopped.accepted)
        writes_after_explicit_zero = len(self.serial.writes)

        watchdog = self.sink.tick(now_monotonic_ns=1_060_000_000)

        self.assertIsNone(watchdog)
        self.assertEqual(len(self.serial.writes), writes_after_explicit_zero)
        self.assertEqual(self.sink.snapshot().active_binding, self.act)

    def test_unsafe_telemetry_immediately_zeros_an_active_policy(self) -> None:
        self.activate(self.act)
        count_before = len(self.serial.writes)
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_040_000_000,
                command_rx_seq=self.packets()[-1]["command_seq"],
                command_valid=True,
                mode=ControlMode.MANUAL_ACTION,
                control_enabled=False,
            )
        )
        self.assertEqual(len(self.serial.writes), count_before + 1)
        packet = self.packets()[-1]
        self.assertEqual(packet["schema_version"], "stm32_manual_command.v1")
        self.assertEqual(packet["Y2"], 0.0)

    def test_unsafe_telemetry_cancels_a_pending_act_to_rl_handoff(self) -> None:
        self.activate(self.act)
        handoff_generation = self.sink.request_handoff(
            self.rl,
            now_monotonic_ns=1_030_000_000,
        )
        writes_before = len(self.serial.writes)

        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_040_000_000,
                command_rx_seq=0,
                command_valid=False,
                mode=ControlMode.MANUAL_ACTION,
                sensor_valid=False,
            )
        )

        snapshot = self.sink.snapshot()
        self.assertGreater(snapshot.generation, handoff_generation)
        self.assertEqual(snapshot.phase.value, "terminal_zero_pending")
        self.assertEqual(snapshot.active_binding, self.act)
        self.assertIsNone(snapshot.target_binding)
        self.assertEqual(len(self.serial.writes), writes_before + 1)
        self.assertEqual(self.packets()[-1]["Y2"], 0.0)

    def test_invalid_input_immediately_zeros_without_waiting_for_state_age(self) -> None:
        self.activate(self.rl)
        count_before = len(self.serial.writes)

        self.sink.invalidate_telemetry(
            receive_monotonic_ns=1_030_000_000,
            stm32_alive=True,
        )

        self.assertEqual(len(self.serial.writes), count_before + 1)
        packet = self.packets()[-1]
        self.assertEqual(packet["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(
            [
                packet[name]
                for name in ("boom_mps", "stick_mps", "bucket_mps", "swing_radps")
            ],
            [0.0] * 4,
        )

    def test_handoff_cannot_overtake_an_in_flight_serial_write(self) -> None:
        generation = self.activate(self.rl)
        self.serial.write_started = threading.Event()
        self.serial.release_write = threading.Event()

        result: list[object] = []

        def submit() -> None:
            result.append(
                self.sink.submit_candidate(
                    candidate(
                        source="rl_follow",
                        generation=generation,
                        mode=ControlMode.VELOCITY_REFERENCE,
                    ),
                    now_monotonic_ns=1_030_000_000,
                )
            )

        submit_thread = threading.Thread(target=submit)
        submit_thread.start()
        self.assertTrue(self.serial.write_started.wait(timeout=1.0))

        handoff_done = threading.Event()

        def handoff() -> None:
            self.sink.request_handoff(self.act, now_monotonic_ns=1_040_000_000)
            handoff_done.set()

        handoff_thread = threading.Thread(target=handoff)
        handoff_thread.start()
        self.assertFalse(handoff_done.wait(timeout=0.05))

        self.serial.release_write.set()
        submit_thread.join(timeout=1.0)
        handoff_thread.join(timeout=1.0)
        self.assertTrue(handoff_done.is_set())

        packets = self.packets()
        self.assertNotEqual(packets[-2]["boom_mps"], 0.0)
        self.assertEqual(packets[-1]["boom_mps"], 0.0)

    def test_terminal_disarm_makes_the_last_write_zero_and_rejects_reactivation(self) -> None:
        generation = self.activate(self.rl)
        moving = self.sink.submit_candidate(
            candidate(
                source="rl_follow",
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
            ),
            now_monotonic_ns=1_030_000_000,
        )
        self.assertTrue(moving.accepted)

        terminal = self.sink.terminal_disarm(now_monotonic_ns=1_040_000_000)
        self.assertTrue(terminal.write_performed)
        self.assertEqual(terminal.reason, "terminal_disarm")
        self.assertEqual(self.packets()[-1]["boom_mps"], 0.0)

        after = self.sink.submit_candidate(
            candidate(
                source="rl_follow",
                generation=generation,
                mode=ControlMode.VELOCITY_REFERENCE,
            ),
            now_monotonic_ns=1_050_000_000,
        )
        self.assertFalse(after.write_performed)
        self.assertEqual(after.reason, "terminally_disarmed")
        with self.assertRaisesRegex(RuntimeError, "terminally disarmed"):
            self.sink.request_handoff(self.act, now_monotonic_ns=1_060_000_000)

    def test_serial_write_failure_latches_the_sink_until_restart(self) -> None:
        self.serial.short_write = True
        with self.assertRaisesRegex(OSError, "short serial write"):
            self.sink.request_handoff(self.act, now_monotonic_ns=1_000_000_000)

        self.serial.short_write = False
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            self.sink.request_handoff(self.rl, now_monotonic_ns=1_010_000_000)

    def test_telemetry_rejects_boolean_command_action_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            telemetry(
                receive_ns=1_000_000_000,
                action=(True, 0.0, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
