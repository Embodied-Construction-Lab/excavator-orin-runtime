import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from edge_runtime.control import ActionSequence, EdgeControlRunner
from edge_runtime.follow import EdgeFollowRuntime
from edge_runtime.kinematics import UrdfBucketTipKinematics
from edge_runtime.trajectory_controller import (
    TrajectoryControlOutput,
    TrajectoryControllerDescriptor,
)
from tests.test_edge_control import RecordingSink, StubRuntime
from tests.test_edge_follow_runtime import machine_profile, mission, trajectory
from tests.test_edge_kinematics import URDF


class EdgeControlTrackingAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.temp_dir.name) / "edge_control.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _records(self):
        return [
            json.loads(line)
            for line in self.audit_path.read_text(encoding="utf-8").splitlines()
        ]

    def _assert_terminal_has_no_policy_identity(self, terminal):
        for sample_only_field in (
            "sample_id",
            "policy_action_seq",
            "action_order",
            "normalized_action",
            "commanded_normalized_action",
            "physical_action",
        ):
            self.assertNotIn(sample_only_field, terminal)

    def test_active_sample_preserves_action_and_tracking_contract(self):
        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
            trace_run_id="follow-run-test",
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
        record = next(
            record
            for record in self._records()
            if record["record_type"] == "policy_sample"
        )
        self.assertEqual(
            record["commanded_normalized_action"],
            [0.08, -0.16, 0.24, -0.32],
        )
        self.assertEqual(record["trajectory_controller_backend"], "test_controller")
        self.assertEqual(record["sample_id"], 0)
        self.assertEqual(record["policy_action_seq"], packet["seq"])
        self.assertEqual(record["trace_semantics"], "commanded_normalized_action")
        self.assertEqual(
            record["action_order"],
            ["boom", "stick", "bucket", "swing"],
        )
        self.assertEqual(record["trace_run_id"], "follow-run-test")
        self.assertEqual(record["record_type"], "policy_sample")
        self.assertEqual(record["status"], "active")
        self.assertEqual(record["result"], "active")
        self.assertEqual(record["waypoint_index"], 1)
        self.assertEqual(record["reference_waypoint_ros_m"], [0.8, -0.1, 0.0])
        self.assertEqual(runner.action_datagrams, 2)

    def test_active_step_without_complete_tracking_context_fails_closed(self):
        cases = (
            ("reference", {"reference_waypoint_ros_m": None}),
            ("index", {"waypoint_index": None}),
        )
        for label, override in cases:
            with self.subTest(label=label):
                class InvalidTrackingRuntime(StubRuntime):
                    def step(self, machine_state, *, now_s):
                        values = vars(super().step(machine_state, now_s=now_s))
                        return SimpleNamespace(**{**values, **override})

                sink = RecordingSink()
                audit_path = self.audit_path.with_name(f"{label}.jsonl")
                runner = EdgeControlRunner(
                    runtime=InvalidTrackingRuntime(),
                    action_sink=sink,
                    audit_path=audit_path,
                    valid_for_ms=300,
                    trace_run_id=f"follow-missing-{label}",
                )

                with self.assertLogs("orin_edge_control", level="WARNING"):
                    step = runner.observe(
                        {"seq": 4, "stamp_ms": 1200},
                        now_s=2.0,
                        action_stamp_ms=5000,
                    )
                runner.close(action_stamp_ms=5001)

                self.assertIsNone(step)
                self.assertEqual(
                    [packet["action"] for packet in sink.payloads],
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                )
                records = [
                    json.loads(line)
                    for line in audit_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(
                    [record["record_type"] for record in records],
                    ["terminal"],
                )
                self.assertEqual(records[0]["result"], "rejected")
                self.assertIn("tracking context invalid", records[0]["reason"])

    def test_terminal_declares_trailing_policy_sample_enqueue_drop(self):
        class DropSecondPolicySampleWriter:
            instance = None

            def __init__(self, _path):
                type(self).instance = self
                self.records = []

            def append(self, record):
                copied = dict(record)
                if (
                    copied.get("record_type") == "policy_sample"
                    and copied.get("sample_id") == 1
                ):
                    return False
                self.records.append(copied)
                return True

            def close(self):
                return True

        with mock.patch(
            "edge_runtime.control._BoundedJsonlAuditWriter",
            DropSecondPolicySampleWriter,
        ):
            runner = EdgeControlRunner(
                runtime=StubRuntime(),
                action_sink=RecordingSink(),
                audit_path=self.audit_path,
                valid_for_ms=300,
                trace_run_id="follow-trailing-drop",
                monotonic_clock=lambda: 2.2,
            )
            runner.observe(
                {"seq": 4, "stamp_ms": 1200},
                now_s=2.0,
                action_stamp_ms=5000,
            )
            runner.observe(
                {"seq": 5, "stamp_ms": 1300},
                now_s=2.1,
                action_stamp_ms=5100,
            )
            runner.close(action_stamp_ms=5101)

        writer = DropSecondPolicySampleWriter.instance
        self.assertIsNotNone(writer)
        terminal = writer.records[-1]
        self.assertEqual(terminal["record_type"], "terminal")
        self.assertEqual(terminal["expected_policy_sample_count"], 2)
        self.assertEqual(terminal["accepted_policy_sample_count"], 1)
        self.assertEqual(terminal["dropped_policy_sample_count"], 1)

    def test_terminal_is_retried_after_policy_sample_enqueue_exception(self):
        class RaiseFirstPolicySampleWriter:
            instance = None

            def __init__(self, _path):
                type(self).instance = self
                self.records = []
                self.raised = False

            def append(self, record):
                copied = dict(record)
                if copied.get("record_type") == "policy_sample" and not self.raised:
                    self.raised = True
                    raise RuntimeError("transient enqueue failure")
                self.records.append(copied)
                return True

            def close(self):
                return True

        with mock.patch(
            "edge_runtime.control._BoundedJsonlAuditWriter",
            RaiseFirstPolicySampleWriter,
        ), self.assertLogs("orin_edge_control", level="ERROR"):
            runner = EdgeControlRunner(
                runtime=StubRuntime(),
                action_sink=RecordingSink(),
                audit_path=self.audit_path,
                valid_for_ms=300,
                trace_run_id="follow-enqueue-exception",
                monotonic_clock=lambda: 2.2,
            )
            runner.observe(
                {"seq": 4, "stamp_ms": 1200},
                now_s=2.0,
                action_stamp_ms=5000,
            )
            runner.close(action_stamp_ms=5001)

        writer = RaiseFirstPolicySampleWriter.instance
        self.assertIsNotNone(writer)
        self.assertEqual([record["record_type"] for record in writer.records], ["terminal"])
        terminal = writer.records[0]
        self.assertEqual(terminal["expected_policy_sample_count"], 1)
        self.assertEqual(terminal["accepted_policy_sample_count"], 0)
        self.assertEqual(terminal["dropped_policy_sample_count"], 1)

    def test_unavailable_audit_path_never_blocks_motion_or_terminal_zero(self):
        blocker = Path(self.temp_dir.name) / "not-a-directory"
        blocker.write_text("occupied", encoding="utf-8")
        sink = RecordingSink()

        with self.assertLogs("orin_audit_writer", level="ERROR"):
            runner = EdgeControlRunner(
                runtime=StubRuntime(),
                action_sink=sink,
                audit_path=blocker / "edge_control.jsonl",
                valid_for_ms=300,
            )
            step = runner.observe(
                {"seq": 4, "stamp_ms": 1200},
                now_s=2.0,
                action_stamp_ms=5000,
            )
            runner.close(action_stamp_ms=5001)

        self.assertIsNotNone(step)
        self.assertEqual(
            [packet["action"] for packet in sink.payloads],
            [[0.01, -0.02, 0.03, -0.04], [0.0, 0.0, 0.0, 0.0]],
        )

    def test_audit_write_failure_is_disabled_without_affecting_motion(self):
        class FailingAudit:
            def write(self, _payload):
                raise OSError("disk unavailable")

            def close(self):
                raise OSError("close unavailable")

        sink = RecordingSink()
        with mock.patch(
            "edge_runtime.audit_writer.Path.open",
            return_value=FailingAudit(),
        ), self.assertLogs("orin_audit_writer", level="ERROR"):
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
            runner.close(action_stamp_ms=5001)

        self.assertIsNotNone(step)
        self.assertEqual(
            [packet["action"] for packet in sink.payloads],
            [[0.01, -0.02, 0.03, -0.04], [0.0, 0.0, 0.0, 0.0]],
        )

    def test_slow_audit_write_does_not_delay_observe(self):
        class BlockingAudit:
            def __init__(self):
                self.write_started = threading.Event()
                self.release_write = threading.Event()

            def write(self, payload):
                self.write_started.set()
                self.release_write.wait(1.0)
                return len(payload)

            def flush(self):
                return None

            def close(self):
                return None

        audit = BlockingAudit()
        sink = RecordingSink()
        with mock.patch(
            "edge_runtime.audit_writer.Path.open",
            return_value=audit,
        ):
            runner = EdgeControlRunner(
                runtime=StubRuntime(),
                action_sink=sink,
                audit_path=self.audit_path,
                valid_for_ms=300,
            )

            started = time.perf_counter()
            runner.observe(
                {"seq": 4, "stamp_ms": 1200},
                now_s=2.0,
                action_stamp_ms=5000,
            )
            first_elapsed_s = time.perf_counter() - started
            self.assertLess(first_elapsed_s, 0.05)
            self.assertTrue(audit.write_started.wait(0.2))

            started = time.perf_counter()
            runner.observe(
                {"seq": 5, "stamp_ms": 1300},
                now_s=2.1,
                action_stamp_ms=5100,
            )
            second_elapsed_s = time.perf_counter() - started
            self.assertLess(second_elapsed_s, 0.05)

            audit.release_write.set()
            runner.close(action_stamp_ms=5101)

        self.assertEqual(
            [packet["action"] for packet in sink.payloads],
            [
                [0.01, -0.02, 0.03, -0.04],
                [0.01, -0.02, 0.03, -0.04],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_consecutive_remote_follow_runners_share_one_action_sequence(self):
        sink = RecordingSink()
        sequence = ActionSequence()
        first = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
            action_sequence=sequence,
        )
        first.observe(
            {"seq": 4, "stamp_ms": 1200},
            now_s=2.0,
            action_stamp_ms=5000,
        )
        first.close(action_stamp_ms=5001)

        second = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
            action_sequence=sequence,
        )
        second.observe(
            {"seq": 5, "stamp_ms": 1300},
            now_s=2.1,
            action_stamp_ms=5100,
        )
        second.close(action_stamp_ms=5101)

        self.assertEqual([packet["seq"] for packet in sink.payloads], [0, 1, 2, 3])
        records = self._records()
        samples = [
            record for record in records if record["record_type"] == "policy_sample"
        ]
        terminals = [
            record for record in records if record["record_type"] == "terminal"
        ]
        self.assertEqual([record["sample_id"] for record in samples], [0, 0])
        self.assertEqual([record["policy_action_seq"] for record in samples], [0, 2])
        trace_run_ids = [record["trace_run_id"] for record in samples]
        self.assertTrue(all(trace_run_ids))
        self.assertNotEqual(trace_run_ids[0], trace_run_ids[1])
        self.assertEqual(len(terminals), 2)
        self.assertEqual(
            [record["trace_run_id"] for record in terminals],
            trace_run_ids,
        )

    def test_trace_run_id_is_stable_within_one_follow_runner(self):
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=RecordingSink(),
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

        runner.observe(
            {"seq": 4, "stamp_ms": 1200},
            now_s=2.0,
            action_stamp_ms=5000,
        )
        runner.observe(
            {"seq": 5, "stamp_ms": 1300},
            now_s=2.1,
            action_stamp_ms=5100,
        )
        runner.close(action_stamp_ms=5101)

        records = self._records()
        samples = [
            record for record in records if record["record_type"] == "policy_sample"
        ]
        terminal = next(
            record for record in records if record["record_type"] == "terminal"
        )
        self.assertEqual([record["sample_id"] for record in samples], [0, 1])
        self.assertTrue(samples[0]["trace_run_id"])
        self.assertEqual(samples[0]["trace_run_id"], samples[1]["trace_run_id"])
        self.assertEqual(terminal["trace_run_id"], samples[0]["trace_run_id"])

    def test_close_appends_one_interrupted_terminal_without_policy_identity(self):
        runner = EdgeControlRunner(
            runtime=StubRuntime(),
            action_sink=RecordingSink(),
            audit_path=self.audit_path,
            valid_for_ms=300,
            trace_run_id="follow-interrupted",
            monotonic_clock=lambda: 2.75,
        )

        runner.observe(
            {"seq": 4, "stamp_ms": 1200},
            now_s=2.0,
            action_stamp_ms=5000,
        )
        runner.close(action_stamp_ms=5001)

        records = self._records()
        self.assertEqual(
            [record["record_type"] for record in records],
            ["policy_sample", "terminal"],
        )
        sample, terminal = records
        self.assertEqual(sample["trace_run_id"], "follow-interrupted")
        self.assertEqual(terminal["trace_run_id"], "follow-interrupted")
        self.assertEqual(terminal["status"], "terminal")
        self.assertEqual(terminal["result"], "interrupted")
        self.assertEqual(terminal["runtime_monotonic_s"], 2.75)
        self.assertEqual(terminal["elapsed_s"], 0.75)
        self.assertEqual(terminal["waypoint_index"], 1)
        self.assertEqual(terminal["reference_waypoint_ros_m"], [0.8, -0.1, 0.0])
        self._assert_terminal_has_no_policy_identity(terminal)

    def test_completed_trajectory_sends_zero_instead_of_last_policy_action(self):
        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=StubRuntime(completed=True),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
            trace_run_id="follow-completed",
        )

        runner.observe(
            {"seq": 5, "stamp_ms": 1300},
            now_s=2.1,
            action_stamp_ms=5100,
        )

        self.assertEqual(sink.payloads[-1]["action"], [0.0, 0.0, 0.0, 0.0])
        runner.close(action_stamp_ms=5101)
        records = self._records()
        self.assertEqual(len(records), 1)
        terminal = records[0]
        self.assertEqual(terminal["record_type"], "terminal")
        self.assertEqual(terminal["status"], "terminal")
        self.assertEqual(terminal["result"], "completed")
        self.assertEqual(terminal["trace_run_id"], "follow-completed")
        self.assertEqual(terminal["runtime_monotonic_s"], 2.1)
        self.assertEqual(terminal["elapsed_s"], 0.0)
        self._assert_terminal_has_no_policy_identity(terminal)

    def test_runtime_failure_and_close_each_send_zero(self):
        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=StubRuntime(failure="sensor_invalid"),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
            trace_run_id="follow-rejected",
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
        records = self._records()
        self.assertEqual(len(records), 1)
        terminal = records[0]
        self.assertEqual(terminal["record_type"], "terminal")
        self.assertEqual(terminal["status"], "terminal")
        self.assertEqual(terminal["result"], "rejected")
        self.assertEqual(terminal["trace_run_id"], "follow-rejected")
        self.assertEqual(terminal["runtime_monotonic_s"], 2.2)
        self.assertEqual(terminal["elapsed_s"], 0.0)
        self._assert_terminal_has_no_policy_identity(terminal)
        self.assertEqual(terminal["trajectory_controller_backend"], "test_controller")

    def test_first_rejection_audits_the_configured_runtime_backend(self):
        class Controller:
            descriptor = TrajectoryControllerDescriptor(
                backend_id="cartesian_p",
                implementation="test.Controller",
            )

            def reset(self):
                return None

            def compute_action(self, _observation):
                return TrajectoryControlOutput(
                    normalized_action=(0.0, 0.0, 0.0, 0.0),
                    inference_ms=0.0,
                )

        urdf_path = Path(self.temp_dir.name) / "machine.urdf"
        urdf_path.write_text(URDF, encoding="utf-8")
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=UrdfBucketTipKinematics.from_path(urdf_path),
            controller=Controller(),
            trajectory=trajectory(),
            mission=mission(),
        )
        runner = EdgeControlRunner(
            runtime=runtime,
            action_sink=RecordingSink(),
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

        result = runner.observe(
            {"seq": 1, "stamp_ms": 1_000},
            now_s=1.0,
            action_stamp_ms=1_000,
        )
        runner.close(action_stamp_ms=1_001)

        self.assertIsNone(result)
        terminal = self._records()[0]
        self.assertEqual(terminal["trajectory_controller_backend"], "cartesian_p")

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
                    follow_elapsed_s=60.0,
                    reference_waypoint_ros_m=(0.8, -0.1, 0.0),
                )

        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=TimeoutRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

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
        runner.close(action_stamp_ms=63001)
        records = self._records()
        self.assertEqual(len(records), 1)
        terminal = records[0]
        self.assertEqual(terminal["record_type"], "terminal")
        self.assertEqual(terminal["status"], "terminal")
        self.assertEqual(terminal["result"], "timeout")
        self.assertEqual(terminal["elapsed_s"], 60.0)
        self.assertEqual(terminal["reference_waypoint_ros_m"], [0.8, -0.1, 0.0])
        self._assert_terminal_has_no_policy_identity(terminal)

    def test_transient_runtime_failure_starts_a_new_trace_on_recovery(self):
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
                    reference_waypoint_ros_m=(0.8, -0.1, 0.0),
                )

        sink = RecordingSink()
        runner = EdgeControlRunner(
            runtime=FlakyRuntime(),
            action_sink=sink,
            audit_path=self.audit_path,
            valid_for_ms=300,
        )

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
        self.assertEqual(sink.payloads[-1]["action"], [0.02, -0.025, 0.0075, -0.15])
        runner.close(action_stamp_ms=65001)
        rejected_record, recovered_record, terminal_record = self._records()
        self.assertEqual(rejected_record["exception_type"], "ValueError")
        self.assertEqual(rejected_record["consecutive_rejections"], 1)
        self._assert_terminal_has_no_policy_identity(rejected_record)
        self.assertEqual(rejected_record["record_type"], "terminal")
        self.assertEqual(rejected_record["result"], "rejected")
        self.assertIn("trace_run_id", rejected_record)
        self.assertEqual(recovered_record["consecutive_rejections"], 0)
        self.assertEqual(recovered_record["runtime_monotonic_s"], 65.0)
        self.assertEqual(recovered_record["sample_id"], 0)
        self.assertEqual(recovered_record["policy_action_seq"], 1)
        self.assertNotEqual(
            recovered_record["trace_run_id"],
            rejected_record["trace_run_id"],
        )
        self.assertIn("loop_elapsed_ms", recovered_record)
        self.assertEqual(terminal_record["record_type"], "terminal")
        self.assertEqual(terminal_record["result"], "interrupted")
        self.assertEqual(
            terminal_record["trace_run_id"],
            recovered_record["trace_run_id"],
        )


if __name__ == "__main__":
    unittest.main()
