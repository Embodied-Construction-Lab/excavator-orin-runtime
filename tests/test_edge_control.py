import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import orin_state_sender

from edge_runtime.control import EdgeControlRunner
from edge_runtime.follow import EdgeFollowStep


class StubRuntime:
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


class EdgeControlRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.temp_dir.name) / "edge_control.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sends_local_physical_action_without_sign_or_scale_changes(self):
        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

        step = runner.observe(
            {"seq": 4, "stamp_ms": 1200},
            now_s=2.0,
            action_stamp_ms=5000,
        )

        self.assertIsNotNone(step)
        self.assertEqual(len(sink.payloads), 1)
        packet = sink.payloads[0]
        self.assertEqual(packet["action_order"], ["boom", "stick", "bucket", "swing"])
        self.assertEqual(packet["action"], [0.01, -0.02, 0.03, -0.04])
        self.assertEqual(packet["stamp_ms"], 5000)
        self.assertEqual(packet["valid_for_ms"], 300)
        self.assertEqual(runner.action_datagrams, 1)
        runner.close(action_stamp_ms=5001)
        self.assertEqual(runner.action_datagrams, 2)

    def test_completed_trajectory_sends_zero_instead_of_last_policy_action(self):
        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=StubRuntime(completed=True),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

        runner.observe(
            {"seq": 5, "stamp_ms": 1300},
            now_s=2.1,
            action_stamp_ms=5100,
        )

        self.assertEqual(sink.payloads[-1]["action"], [0.0, 0.0, 0.0, 0.0])
        runner.close(action_stamp_ms=5101)

    def test_runtime_failure_and_close_each_send_zero(self):
        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=StubRuntime(failure="sensor_invalid"),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

        result = runner.observe(
            {"seq": 6, "stamp_ms": 1400},
            now_s=2.2,
            action_stamp_ms=5200,
        )
        runner.close(action_stamp_ms=5300)

        self.assertIsNone(result)
        self.assertEqual(
            [packet["action"] for packet in sink.payloads],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        )

    def test_timeout_result_sends_zero_and_never_reuses_nonzero_action(self):
        class TimeoutRuntime:
            def step(self, machine_state, *, now_s):
                return SimpleNamespace(
                    source_seq=machine_state["seq"],
                    source_stamp_ms=machine_state["stamp_ms"],
                    waypoint_index=1,
                    completed=False,
                    result="TIMEOUT",
                    bucket_tip_ros_m=(0.1, 0.2, 0.3),
                    normalized_action=(0.5, -0.5, 0.25, -0.25),
                    physical_action=(0.02, -0.025, 0.0075, -0.15),
                )

        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=TimeoutRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )
        self.addCleanup(runner.close, action_stamp_ms=63001)

        runner.observe(
            {"seq": 7, "stamp_ms": 1500},
            now_s=62.0,
            action_stamp_ms=62000,
        )
        runner.observe(
            {"seq": 8, "stamp_ms": 1600},
            now_s=63.0,
            action_stamp_ms=63000,
        )

        self.assertEqual(sink.payloads[-2]["action"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(sink.payloads[-1]["action"], [0.0, 0.0, 0.0, 0.0])
        records = [
            json.loads(line)
            for line in self.audit_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[-1]["status"], "TIMEOUT")

    def test_transient_runtime_failure_sends_zero_then_recovers_on_valid_state(self):
        class FlakyRuntime:
            def __init__(self):
                self.calls = 0

            def step(self, machine_state, *, now_s):
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("temporary_state_loss")
                return SimpleNamespace(
                    source_seq=machine_state["seq"],
                    source_stamp_ms=machine_state["stamp_ms"],
                    waypoint_index=1,
                    completed=False,
                    result="ACTIVE",
                    episode_progress=0.1,
                    waypoint_distance_m=0.4,
                    bucket_tip_ros_m=(0.1, 0.2, 0.3),
                    normalized_action=(0.5, -0.5, 0.25, -0.25),
                    physical_action=(0.02, -0.025, 0.0075, -0.15),
                )

        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=FlakyRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )
        self.addCleanup(runner.close, action_stamp_ms=65001)

        rejected = runner.observe(
            {"seq": 9, "stamp_ms": 1700},
            now_s=64.0,
            action_stamp_ms=64000,
        )
        recovered = runner.observe(
            {"seq": 10, "stamp_ms": 1800},
            now_s=65.0,
            action_stamp_ms=65000,
        )

        self.assertIsNone(rejected)
        self.assertIsNotNone(recovered)
        self.assertEqual(sink.payloads[-2]["action"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            sink.payloads[-1]["action"],
            [0.02, -0.025, 0.0075, -0.15],
        )
        records = [
            json.loads(line)
            for line in self.audit_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[-2]["exception_type"], "ValueError")
        self.assertEqual(records[-2]["consecutive_rejections"], 1)
        self.assertEqual(records[-1]["consecutive_rejections"], 0)
        self.assertEqual(records[-1]["runtime_monotonic_s"], 65.0)
        self.assertIn("loop_elapsed_ms", records[-1])

    def test_loopback_control_uses_existing_action_relay_as_only_serial_writer(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
            while (
                time.monotonic() < deadline
                and not any(b";0.01;-0.02;0.03;-0.04\n" in value for value in serial.writes)
            ):
                time.sleep(0.005)
            self.assertTrue(
                any(b";0.01;-0.02;0.03;-0.04\n" in value for value in serial.writes)
            )
        finally:
            runner.close(action_stamp_ms=orin_state_sender.now_ms())
            relay.close()
            sender.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
