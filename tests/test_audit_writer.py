import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from edge_runtime.audit_writer import _BoundedJsonlAuditWriter


class RecordingHandle:
    def __init__(self):
        self.lines = []
        self.flush_count = 0
        self.closed = False

    def write(self, payload):
        self.lines.append(payload)
        return len(payload)

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.closed = True


class BlockingHandle(RecordingHandle):
    def __init__(self):
        super().__init__()
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def write(self, payload):
        self.write_started.set()
        self.release_write.wait(1.0)
        return super().write(payload)


class BoundedJsonlAuditWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audit_path = Path(self.temp_dir.name) / "audit.jsonl"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_slow_disk_never_blocks_the_producer_and_close_is_bounded(self):
        handle = BlockingHandle()
        writer = _BoundedJsonlAuditWriter(
            self.audit_path,
            max_queue_size=4,
            close_timeout_s=0.02,
            open_file=lambda _path: handle,
        )
        self.assertTrue(writer.append({"record_type": "policy_sample", "seq": 1}))
        self.assertTrue(handle.write_started.wait(0.2))

        started = time.perf_counter()
        self.assertTrue(writer.append({"record_type": "policy_sample", "seq": 2}))
        append_elapsed_s = time.perf_counter() - started
        self.assertLess(append_elapsed_s, 0.05)

        started = time.perf_counter()
        with self.assertLogs("orin_audit_writer", level="ERROR") as captured:
            self.assertFalse(writer.close())
        close_elapsed_s = time.perf_counter() - started
        self.assertLess(close_elapsed_s, 0.1)
        self.assertIn("drain timed out", captured.output[0])

        handle.release_write.set()
        self.assertTrue(writer.close())

    def test_queue_full_drops_active_sample_but_reserves_terminal_slot(self):
        handle = BlockingHandle()
        writer = _BoundedJsonlAuditWriter(
            self.audit_path,
            max_queue_size=2,
            close_timeout_s=0.5,
            open_file=lambda _path: handle,
        )
        self.assertTrue(writer.append({"record_type": "policy_sample", "seq": 1}))
        self.assertTrue(handle.write_started.wait(0.2))
        self.assertTrue(writer.append({"record_type": "policy_sample", "seq": 2}))
        self.assertFalse(writer.append({"record_type": "policy_sample", "seq": 3}))
        self.assertTrue(
            writer.append(
                {
                    "record_type": "terminal",
                    "status": "terminal",
                    "result": "completed",
                }
            )
        )

        handle.release_write.set()
        with self.assertLogs("orin_audit_writer", level="WARNING") as captured:
            self.assertTrue(writer.close())

        records = [json.loads(line) for line in handle.lines]
        self.assertEqual([record.get("seq") for record in records[:-1]], [1, 2])
        self.assertEqual(records[-1]["record_type"], "terminal")
        self.assertEqual(writer.dropped_record_count, 1)
        self.assertIn("dropped 1 record", captured.output[0])

    def test_open_failure_disables_audit_without_raising_to_producer(self):
        def fail_open(_path):
            raise OSError("filesystem unavailable")

        with self.assertLogs("orin_audit_writer", level="ERROR"):
            writer = _BoundedJsonlAuditWriter(
                self.audit_path,
                open_file=fail_open,
            )
            self.assertTrue(writer.wait_until_disabled(0.2))
            self.assertFalse(writer.append({"record_type": "policy_sample"}))
            self.assertTrue(writer.close())

    def test_serialization_failure_disables_audit_in_background(self):
        handle = RecordingHandle()
        writer = _BoundedJsonlAuditWriter(
            self.audit_path,
            open_file=lambda _path: handle,
        )

        with self.assertLogs("orin_audit_writer", level="ERROR"):
            self.assertTrue(writer.append({"bad": object()}))
            self.assertTrue(writer.wait_until_disabled(0.2))
            self.assertFalse(writer.append({"record_type": "policy_sample"}))
            self.assertTrue(writer.close())

    def test_write_and_flush_failures_never_escape_the_worker(self):
        class WriteFailureHandle(RecordingHandle):
            def write(self, _payload):
                raise OSError("disk write failed")

        write_handle = WriteFailureHandle()
        with self.assertLogs("orin_audit_writer", level="ERROR"):
            writer = _BoundedJsonlAuditWriter(
                self.audit_path,
                open_file=lambda _path: write_handle,
            )
            self.assertTrue(writer.append({"record_type": "policy_sample"}))
            self.assertTrue(writer.wait_until_disabled(0.2))
            self.assertTrue(writer.close())

        class FlushFailureHandle(RecordingHandle):
            def flush(self):
                raise OSError("disk flush failed")

        flush_handle = FlushFailureHandle()
        with self.assertLogs("orin_audit_writer", level="ERROR"):
            writer = _BoundedJsonlAuditWriter(
                self.audit_path,
                open_file=lambda _path: flush_handle,
            )
            self.assertTrue(writer.append({"record_type": "terminal"}))
            self.assertTrue(writer.close())
            self.assertTrue(writer.disabled)

    def test_normal_close_drains_and_flushes_terminal_record(self):
        writer = _BoundedJsonlAuditWriter(
            self.audit_path,
            max_queue_size=4,
            close_timeout_s=0.5,
        )
        self.assertTrue(
            writer.append(
                {
                    "record_type": "policy_sample",
                    "trace_run_id": "trace-1",
                }
            )
        )
        self.assertTrue(
            writer.append(
                {
                    "record_type": "terminal",
                    "status": "terminal",
                    "result": "completed",
                    "trace_run_id": "trace-1",
                }
            )
        )

        self.assertTrue(writer.close())

        records = [
            json.loads(line)
            for line in self.audit_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["record_type"] for record in records], [
            "policy_sample",
            "terminal",
        ])
        self.assertEqual(records[-1]["result"], "completed")


if __name__ == "__main__":
    unittest.main()
