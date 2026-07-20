import importlib.util
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "orin_csv_replay.py"
SPEC = importlib.util.spec_from_file_location("orin_csv_replay", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


class OrinCsvReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profile = self.root / "machine_profile.json"
        self.profile.write_text(
            json.dumps(
                {
                    "action_order": ["boom", "stick", "bucket", "swing"],
                    "actuators": {
                        "boom": {"max_speed_positive": 0.0351, "max_speed_negative": 0.0185},
                        "stick": {"max_speed_positive": 0.0444, "max_speed_negative": 0.0357},
                        "bucket": {"max_speed_positive": 0.0342, "max_speed_negative": 0.0419},
                        "swing": {"max_speed_positive": 0.6, "max_speed_negative": 0.6},
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, rows):
        path = self.root / "replay.csv"
        header = (
            "sample_index,timestamp_s,boom_v_ref_mps,stick_v_ref_mps,"
            "bucket_v_ref_mps,swing_v_ref_radps\n"
        )
        path.write_text(
            "# units: boom/stick/bucket velocity=m/s, swing velocity=rad/s\n"
            "# profile_action_order=boom|stick|bucket|swing\n"
            + header
            + "".join(rows),
            encoding="utf-8",
        )
        return path

    def test_loads_physical_velocity_columns_without_rescaling(self):
        csv_path = self.write_csv(
            [
                "0,0.0,0.01755,-0.01785,0.01,-0.3\n",
                "1,0.05,0,0,0,0\n",
            ]
        )

        samples = replay.load_replay(csv_path, self.profile, max_duration_s=10.0)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].action, (0.01755, -0.01785, 0.01, -0.3))
        self.assertEqual(samples[1].action, (0.0, 0.0, 0.0, 0.0))

    def test_rejects_velocity_outside_machine_profile(self):
        csv_path = self.write_csv(
            [
                "0,0.0,0.0352,0,0,0\n",
                "1,0.05,0,0,0,0\n",
            ]
        )

        with self.assertRaisesRegex(replay.ReplayValidationError, "boom velocity"):
            replay.load_replay(csv_path, self.profile, max_duration_s=10.0)

    def test_policy_packet_preserves_physical_velocity_and_order(self):
        packet = replay.make_policy_action(
            sequence=42,
            action=(0.01755, -0.01785, 0.01, -0.3),
            stamp_ms=123456,
            valid_for_ms=100,
        )

        self.assertEqual(packet["action_order"], ["boom", "stick", "bucket", "swing"])
        self.assertEqual(packet["action"], [0.01755, -0.01785, 0.01, -0.3])
        self.assertEqual(packet["stamp_ms"], 123456)
        self.assertEqual(packet["valid_for_ms"], 100)

    def test_replay_uses_csv_timing_and_appends_zero_tail(self):
        now = [10.0]
        sent = []

        def sleep(duration):
            now[0] += duration

        samples = (
            replay.ReplaySample(0.0, (0.01, 0.0, 0.0, 0.0)),
            replay.ReplaySample(0.05, (0.0, 0.0, 0.0, 0.0)),
        )

        count = replay.replay_samples(
            samples,
            sent.append,
            valid_for_ms=100,
            zero_tail_count=3,
            zero_interval_s=0.01,
            max_lag_s=0.02,
            monotonic=lambda: now[0],
            wall_ms=lambda: int(now[0] * 1000),
            sleep=sleep,
            sequence_start=100,
        )

        self.assertEqual(count, 2)
        self.assertEqual([packet["seq"] for packet in sent], [100, 101, 102, 103, 104])
        self.assertEqual(sent[0]["action"], [0.01, 0.0, 0.0, 0.0])
        self.assertTrue(all(packet["action"] == [0.0] * 4 for packet in sent[1:]))
        self.assertAlmostEqual(now[0], 10.07)

    def test_scheduler_lag_aborts_and_still_sends_zero_tail(self):
        now = [5.0]
        sent = []

        def late_sleep(duration):
            now[0] += duration + 0.1

        samples = (
            replay.ReplaySample(0.0, (0.01, 0.0, 0.0, 0.0)),
            replay.ReplaySample(0.05, (0.01, 0.0, 0.0, 0.0)),
        )

        with self.assertRaisesRegex(RuntimeError, "scheduler lag"):
            replay.replay_samples(
                samples,
                sent.append,
                valid_for_ms=100,
                zero_tail_count=3,
                zero_interval_s=0.0,
                max_lag_s=0.02,
                monotonic=lambda: now[0],
                wall_ms=lambda: int(now[0] * 1000),
                sleep=late_sleep,
                sequence_start=200,
            )

        self.assertEqual(sent[0]["action"], [0.01, 0.0, 0.0, 0.0])
        self.assertTrue(all(packet["action"] == [0.0] * 4 for packet in sent[1:]))

    def test_live_send_requires_exact_motion_authorization(self):
        self.assertFalse(replay.is_execution_authorized(None))
        self.assertFalse(replay.is_execution_authorized("yes"))
        self.assertTrue(replay.is_execution_authorized("ALLOW_CSV_REPLAY"))

    def test_rejects_csv_without_terminal_zero(self):
        csv_path = self.write_csv(["0,0.0,0.01,0,0,0\n"])

        with self.assertRaisesRegex(replay.ReplayValidationError, "explicit zero"):
            replay.load_replay(csv_path, self.profile, max_duration_s=10.0)

    def test_rejects_timestamp_regression(self):
        csv_path = self.write_csv(
            [
                "0,0.1,0.01,0,0,0\n",
                "1,0.05,0,0,0,0\n",
            ]
        )

        with self.assertRaisesRegex(replay.ReplayValidationError, "must not decrease"):
            replay.load_replay(csv_path, self.profile, max_duration_s=10.0)

    def test_authorized_cli_sends_csv_rows_and_zero_tail_over_loopback(self):
        csv_path = self.write_csv(
            [
                "0,0.0,0.01,0,0,0\n",
                "1,0.0,0,0,0,0\n",
            ]
        )
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(receiver.close)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.05)

        result = replay.main(
            [
                str(csv_path),
                "--machine-profile",
                str(self.profile),
                "--target-port",
                str(receiver.getsockname()[1]),
                "--zero-tail-count",
                "2",
                "--zero-interval-ms",
                "0",
                "--print-every",
                "0",
                "--motion-authorization",
                "ALLOW_CSV_REPLAY",
            ]
        )
        packets = []
        while True:
            try:
                payload, _ = receiver.recvfrom(8192)
            except socket.timeout:
                break
            packets.append(json.loads(payload))

        self.assertEqual(result, 0)
        self.assertEqual(len(packets), 4)
        self.assertEqual(packets[0]["action"], [0.01, 0.0, 0.0, 0.0])
        self.assertTrue(all(packet["action"] == [0.0] * 4 for packet in packets[1:]))


if __name__ == "__main__":
    unittest.main()
