import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from edge_runtime.resident_action_audit import AsyncResidentActionAudit


class BlockingTextHandle:
    def __init__(self) -> None:
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def write(self, _value: str) -> int:
        self.write_started.set()
        self.release_write.wait(timeout=1.0)
        return 1

    def flush(self) -> None:
        return None


class AsyncResidentActionAuditTest(unittest.TestCase):
    def test_persists_one_event_without_blocking_the_caller_on_file_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "resident-actions.jsonl"
            audit = AsyncResidentActionAudit(
                path,
                wall_time_ns=lambda: 123_000_000,
            )

            accepted = audit.emit(
                "command_write",
                monotonic_ns=456_000_000,
                source="act_dig",
                action=[0.1, -0.2, 0.3, 0.0],
            )
            audit.close(timeout_s=1.0)

            self.assertTrue(accepted)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {
                    "action": [0.1, -0.2, 0.3, 0.0],
                    "event_type": "command_write",
                    "monotonic_ns": 456_000_000,
                    "schema_version": "resident_action_audit.v1",
                    "source": "act_dig",
                    "wall_time_ns": 123_000_000,
                },
            )

    def test_unwritable_destination_is_reported_without_escaping_close(self) -> None:
        audit = AsyncResidentActionAudit(
            Path("/dev/null/resident-actions.jsonl"),
        )
        for _ in range(100):
            if not audit.status().writer_alive:
                break
            time.sleep(0.001)

        accepted = audit.emit("command_write", monotonic_ns=1)
        audit.close(timeout_s=1.0)

        status = audit.status()
        self.assertFalse(accepted)
        self.assertTrue(status.closed)
        self.assertEqual(status.dropped_event_count, 1)
        self.assertGreaterEqual(status.error_count, 1)

    def test_full_queue_drops_evidence_instead_of_blocking_control(self) -> None:
        handle = BlockingTextHandle()
        audit = AsyncResidentActionAudit(
            Path("/tmp/resident-actions-unused.jsonl"),
            queue_capacity=1,
            open_append=lambda _path: handle,
        )
        self.assertTrue(audit.emit("command_write", monotonic_ns=1))
        self.assertTrue(handle.write_started.wait(timeout=1.0))
        self.assertTrue(audit.emit("command_write", monotonic_ns=2))

        started = time.monotonic()
        accepted = audit.emit("command_write", monotonic_ns=3)
        elapsed_s = time.monotonic() - started

        self.assertFalse(accepted)
        self.assertLess(elapsed_s, 0.05)
        self.assertEqual(audit.status().dropped_event_count, 1)
        handle.release_write.set()
        audit.close(timeout_s=1.0)

    def test_close_remains_bounded_when_the_queue_and_writer_are_busy(self) -> None:
        handle = BlockingTextHandle()
        audit = AsyncResidentActionAudit(
            Path("/tmp/resident-actions-close-unused.jsonl"),
            queue_capacity=1,
            open_append=lambda _path: handle,
        )
        self.assertTrue(audit.emit("command_write", monotonic_ns=1))
        self.assertTrue(handle.write_started.wait(timeout=1.0))
        self.assertTrue(audit.emit("command_write", monotonic_ns=2))

        started = time.monotonic()
        audit.close(timeout_s=0.01)
        elapsed_s = time.monotonic() - started
        handle.release_write.set()
        audit.close(timeout_s=1.0)

        status = audit.status()
        self.assertLess(elapsed_s, 0.05)
        self.assertFalse(status.writer_alive)
        self.assertEqual(status.error_count, 0)
        self.assertEqual(status.written_event_count, 2)


if __name__ == "__main__":
    unittest.main()
