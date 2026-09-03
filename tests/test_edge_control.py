import json
import socket
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import orin_state_sender

from edge_runtime import control
from edge_runtime.control import EdgeControlRunner, FixedActionControlRunner
from edge_runtime.follow import EdgeFollowStep
from edge_runtime.resident_ingress import ResidentVelocityActionAdapter
from edge_runtime.resident_motion import ControlMode, ZERO_ACTION
from edge_runtime.resident_sink import ResidentCommandSink, ResidentTelemetry


class StubRuntime:
    trajectory_controller_backend = "test_controller"

    def __init__(self, *, completed=False, failure=None):
        self.completed = completed
        self.failure = failure

    def step(self, machine_state, *, now_s):
        if self.failure is not None:
            raise ValueError(self.failure)
        return EdgeFollowStep(
            source_seq=machine_state["seq"],
            source_stamp_ms=machine_state["stamp_ms"],
            waypoint_index=1,
            completed=self.completed,
            bucket_tip_ros_m=(0.1, 0.2, 0.3),
            bucket_pitch_rad=0.4,
            observation=tuple(0.0 for _ in range(38)),
            normalized_action=(0.1, -0.2, 0.3, -0.4),
            physical_action=(0.01, -0.02, 0.03, -0.04),
            commanded_normalized_action=(0.08, -0.16, 0.24, -0.32),
            trajectory_controller_backend="test_controller",
            reference_waypoint_ros_m=(0.8, -0.1, 0.0),
        )


class ZeroActionRuntime(StubRuntime):
    def step(self, machine_state, *, now_s):
        return replace(
            super().step(machine_state, now_s=now_s),
            normalized_action=(0.0, 0.0, 0.0, 0.0),
            physical_action=(0.0, 0.0, 0.0, 0.0),
            commanded_normalized_action=(0.0, 0.0, 0.0, 0.0),
        )


class RecordingSink:
    def __init__(self):
        self.payloads = []

    def send(self, payload):
        self.payloads.append(json.loads(payload.decode("utf-8")))


class RecordingSerial:
    def __init__(self):
        self.writes = []
        self.written = threading.Event()

    def write(self, payload):
        self.writes.append(payload)
        self.written.set()
        return len(payload)

    def flush(self):
        return None


def resident_telemetry(
    *,
    receive_ns: int,
    command_seq: int,
    valid: bool,
    mode: ControlMode | None,
) -> ResidentTelemetry:
    return ResidentTelemetry(
        receive_monotonic_ns=receive_ns,
        command_rx_seq=command_seq,
        command_valid=valid,
        command_timed_out=False,
        control_mode=mode,
        command_action=ZERO_ACTION,
        control_enabled=True,
        estop=False,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
    )


class EdgeControlRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.temp_dir.name) / "edge_control.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fixed_action_runner_uses_shared_sequence_and_terminal_zero(self):
        class FixedRuntime:
            def __init__(self):
                self.calls = 0

            def step(self, machine_state, *, now_s):
                self.calls += 1
                return SimpleNamespace(
                    phase="running" if self.calls == 1 else "done",
                    step_index=0,
                    step_label="open_bucket",
                    max_error=0.5 if self.calls == 1 else 0.0,
                    normalized_action=(
                        (0.0, 0.0, 0.6, 0.0)
                        if self.calls == 1
                        else (0.0, 0.0, 0.0, 0.0)
                    ),
                    physical_action=(
                        (0.0, 0.0, 0.02, 0.0)
                        if self.calls == 1
                        else (0.0, 0.0, 0.0, 0.0)
                    ),
                    result="ACTIVE" if self.calls == 1 else "COMPLETED",
                    reason_code="" if self.calls == 1 else "SEQUENCE_COMPLETED",
                )

        sink = RecordingSink()
        sequence = control.ActionSequence(start=20)
        runner = FixedActionControlRunner(
            runtime=FixedRuntime(),
            behavior="ExecuteDump",
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
            action_sequence=sequence,
        )

        active = runner.observe(
            {"seq": 1, "stamp_ms": 1000},
            now_s=1.0,
            action_stamp_ms=2000,
        )
        completed = runner.observe(
            {"seq": 2, "stamp_ms": 1100},
            now_s=1.1,
            action_stamp_ms=2100,
        )
        runner.close(action_stamp_ms=2200)

        self.assertEqual(active.result, "ACTIVE")
        self.assertEqual(completed.result, "COMPLETED")
        self.assertEqual(
            [packet["seq"] for packet in sink.payloads],
            [20, 21, 22],
        )
        self.assertEqual(
            [packet["action"] for packet in sink.payloads],
            [
                [0.0, 0.0, 0.02, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_loopback_control_uses_existing_action_relay_as_only_serial_writer(self):
        try:
            receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except PermissionError:
            self.skipTest("sandbox does not permit local UDP sockets")
        receiver.bind(("127.0.0.1", 0))
        receiver.setblocking(False)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.connect(receiver.getsockname())
        serial = RecordingSerial()
        relay = orin_state_sender.ActionRelay(
            action_sock=receiver,
            ser=serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=sender,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )
        try:
            runner.observe(
                {"seq": 9, "stamp_ms": 1600},
                now_s=2.4,
                action_stamp_ms=orin_state_sender.now_ms(),
            )

            self.assertTrue(serial.written.wait(0.1))
            deadline = time.monotonic() + 0.1
            def has_expected_velocity_command():
                return any(
                    (
                        command["boom_mps"],
                        command["stick_mps"],
                        command["bucket_mps"],
                        command["swing_radps"],
                    )
                    == (0.01, -0.02, 0.03, -0.04)
                    for command in (
                        json.loads(value.decode("ascii")) for value in serial.writes
                    )
                )
            while (
                time.monotonic() < deadline
                and not has_expected_velocity_command()
            ):
                time.sleep(0.005)
            self.assertTrue(has_expected_velocity_command())
        finally:
            runner.close(action_stamp_ms=orin_state_sender.now_ms())
            relay.close()
            sender.close()
            receiver.close()

    def test_resident_control_runner_claims_then_moves_then_requests_stop(self):
        serial = RecordingSerial()
        sink = ResidentCommandSink(serial, max_state_age_ms=200.0)
        sink.initialize(
            resident_telemetry(
                receive_ns=990_000_000,
                command_seq=0,
                valid=False,
                mode=None,
            )
        )
        adapter = ResidentVelocityActionAdapter(
            sink,
            source="rl_follow",
        )
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=adapter,
            audit_path=self.audit_path,
            valid_for_ms=100,
        )

        first_now_ns = time.monotonic_ns()
        first = runner.observe(
            {"seq": 1, "stamp_ms": 9_990},
            now_s=1.0,
            action_stamp_ms=time.time_ns() // 1_000_000,
        )
        claim_packet = json.loads(serial.writes[-1].decode("ascii"))
        sink.observe_telemetry(
            resident_telemetry(
                receive_ns=max(time.monotonic_ns(), first_now_ns + 1),
                command_seq=claim_packet["command_seq"],
                valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )
        second = runner.observe(
            {"seq": 2, "stamp_ms": 10_090},
            now_s=1.03,
            action_stamp_ms=time.time_ns() // 1_000_000,
        )
        runner.close(action_stamp_ms=time.time_ns() // 1_000_000)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        packets = [json.loads(payload.decode("ascii")) for payload in serial.writes]
        self.assertEqual(packets[0]["boom_mps"], 0.0)
        self.assertEqual(
            (
                packets[1]["boom_mps"],
                packets[1]["stick_mps"],
                packets[1]["bucket_mps"],
                packets[1]["swing_radps"],
            ),
            (0.01, -0.02, 0.03, -0.04),
        )
        self.assertEqual(packets[-1]["boom_mps"], 0.0)

    def test_resident_control_failure_before_activation_does_not_raise_or_claim(self):
        serial = RecordingSerial()
        sink = ResidentCommandSink(serial, max_state_age_ms=200.0)
        sink.initialize(
            resident_telemetry(
                receive_ns=990_000_000,
                command_seq=0,
                valid=False,
                mode=None,
            )
        )
        adapter = ResidentVelocityActionAdapter(
            sink,
            source="rl_follow",
        )
        runner = EdgeControlRunner(
            runtime=StubRuntime(failure="sensor_invalid"),
            action_sink=adapter,
            audit_path=self.audit_path,
            valid_for_ms=100,
        )

        result = runner.observe(
            {"seq": 3, "stamp_ms": 10_000},
            now_s=1.0,
            action_stamp_ms=10_000,
        )
        runner.close(action_stamp_ms=10_001)

        self.assertIsNone(result)
        self.assertEqual(serial.writes, [])

    def test_active_zero_action_does_not_release_and_reclaim_resident_policy(self):
        serial = RecordingSerial()
        sink = ResidentCommandSink(serial, max_state_age_ms=200.0)
        sink.initialize(
            resident_telemetry(
                receive_ns=990_000_000,
                command_seq=0,
                valid=False,
                mode=None,
            )
        )
        adapter = ResidentVelocityActionAdapter(
            sink,
            source="rl_follow",
        )
        runner = EdgeControlRunner(
            runtime=ZeroActionRuntime(),
            action_sink=adapter,
            audit_path=self.audit_path,
            valid_for_ms=100,
        )

        runner.observe(
            {"seq": 1, "stamp_ms": 9_990},
            now_s=1.0,
            action_stamp_ms=time.time_ns() // 1_000_000,
        )
        generation = adapter.generation
        claim = json.loads(serial.writes[-1].decode("ascii"))
        sink.observe_telemetry(
            resident_telemetry(
                receive_ns=time.monotonic_ns(),
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )

        runner.observe(
            {"seq": 2, "stamp_ms": 10_090},
            now_s=1.1,
            action_stamp_ms=time.time_ns() // 1_000_000,
        )

        self.assertEqual(adapter.generation, generation)
        self.assertEqual(sink.snapshot().active_binding.source, "rl_follow")
        self.assertEqual(len(serial.writes), 2)
        runner.close(action_stamp_ms=time.time_ns() // 1_000_000)

    def test_shared_resident_runner_close_keeps_rl_authority_and_writes_zero(self):
        serial = RecordingSerial()
        sink = ResidentCommandSink(serial, max_state_age_ms=200.0)
        sink.initialize(
            resident_telemetry(
                receive_ns=990_000_000,
                command_seq=0,
                valid=False,
                mode=None,
            )
        )
        adapter = ResidentVelocityActionAdapter(sink, source="rl_follow")
        generation = adapter.begin_activation(now_monotonic_ns=time.monotonic_ns())
        claim = json.loads(serial.writes[-1].decode("ascii"))
        sink.observe_telemetry(
            resident_telemetry(
                receive_ns=time.monotonic_ns(),
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=adapter,
            audit_path=self.audit_path,
            valid_for_ms=100,
            retain_action_authority=True,
        )
        runner.observe(
            {"seq": 1, "stamp_ms": 9_990},
            now_s=1.0,
            action_stamp_ms=time.time_ns() // 1_000_000,
        )

        runner.close(action_stamp_ms=time.time_ns() // 1_000_000)

        snapshot = sink.snapshot()
        self.assertEqual(snapshot.generation, generation)
        self.assertEqual(snapshot.active_binding.source, "rl_follow")
        terminal = json.loads(serial.writes[-1].decode("ascii"))
        self.assertEqual(
            [
                terminal[name]
                for name in ("boom_mps", "stick_mps", "bucket_mps", "swing_radps")
            ],
            [0.0] * 4,
        )


if __name__ == "__main__":
    unittest.main()
