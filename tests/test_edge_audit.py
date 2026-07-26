import json
import tempfile
import unittest
from pathlib import Path

from edge_runtime.audit import load_audit_records, summarize_audit_records


class EdgeAuditSummaryTest(unittest.TestCase):
    def test_reports_timing_sequence_progress_rejections_and_terminal_zero(self):
        records = [
            {
                "mode": "shadow",
                "status": "ACTIVE",
                "source_seq": 10,
                "source_stamp_ms": 1000,
                "runtime_monotonic_s": 1.0,
                "inference_ms": 0.10,
                "loop_elapsed_ms": 0.30,
                "episode_progress": 0.0,
                "consecutive_rejections": 0,
                "physical_action": [0.01, 0.0, 0.0, 0.0],
            },
            {
                "mode": "shadow",
                "status": "rejected",
                "source_seq": 11,
                "source_stamp_ms": 1100,
                "runtime_monotonic_s": 1.1,
                "loop_elapsed_ms": 0.20,
                "consecutive_rejections": 1,
            },
            {
                "mode": "shadow",
                "status": "TIMEOUT",
                "source_seq": 13,
                "source_stamp_ms": 1300,
                "runtime_monotonic_s": 1.3,
                "inference_ms": 0.0,
                "loop_elapsed_ms": 0.25,
                "episode_progress": 1.0,
                "consecutive_rejections": 0,
                "physical_action": [0.0, 0.0, 0.0, 0.0],
            },
        ]

        summary = summarize_audit_records(records)

        self.assertEqual(summary["record_count"], 3)
        self.assertEqual(summary["status_counts"], {"ACTIVE": 1, "rejected": 1, "TIMEOUT": 1})
        self.assertEqual(summary["source_sequence"]["missing_count"], 1)
        self.assertEqual(summary["source_sequence"]["non_increasing_count"], 0)
        self.assertEqual(summary["progress"]["monotonic_violations"], 0)
        self.assertEqual(summary["max_consecutive_rejections"], 1)
        self.assertTrue(summary["timeout_seen"])
        self.assertTrue(summary["final_physical_action_zero"])
        self.assertAlmostEqual(summary["state_input_hz"], 1000.0 / 150.0)
        self.assertAlmostEqual(summary["inference_ms"]["max"], 0.1)
        self.assertAlmostEqual(summary["loop_elapsed_ms"]["max"], 0.3)

    def test_loader_reports_the_bad_jsonl_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            path.write_text(
                json.dumps({"status": "ACTIVE"}) + "\nnot-json\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_audit_records(path)


if __name__ == "__main__":
    unittest.main()
