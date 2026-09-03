"""Best-effort asynchronous JSONL audit output for motion runtimes."""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TextIO


LOGGER = logging.getLogger("orin_audit_writer")


class _BoundedJsonlAuditWriter:
    """Keep all filesystem work off the motion thread.

    One queue slot is reserved for a terminal record. Active samples are
    best-effort and may be dropped under sustained disk backpressure; the
    caller never waits for disk I/O.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_queue_size: int = 1024,
        close_timeout_s: float = 1.0,
        open_file: Optional[Callable[[Path], TextIO]] = None,
    ) -> None:
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size < 2
        ):
            raise ValueError("audit max_queue_size must be an integer >= 2")
        if (
            isinstance(close_timeout_s, bool)
            or not isinstance(close_timeout_s, (int, float))
            or close_timeout_s <= 0.0
        ):
            raise ValueError("audit close_timeout_s must be positive")
        if open_file is not None and not callable(open_file):
            raise ValueError("audit open_file must be callable")
        self._path = Path(path)
        self._max_queue_size = max_queue_size
        self._close_timeout_s = float(close_timeout_s)
        self._open_file = open_file or self._default_open
        self._queue: queue.Queue[Mapping[str, Any]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._close_requested = threading.Event()
        self._disabled = threading.Event()
        self._dropped_record_count = 0
        self._drop_reported = False
        self._thread = threading.Thread(
            target=self._run,
            name="orin-jsonl-audit-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def disabled(self) -> bool:
        return self._disabled.is_set()

    @property
    def dropped_record_count(self) -> int:
        return self._dropped_record_count

    def wait_until_disabled(self, timeout_s: float) -> bool:
        return self._disabled.wait(timeout_s)

    def append(self, record: Mapping[str, Any]) -> bool:
        """Enqueue without waiting; return whether this record was accepted."""
        if self._close_requested.is_set() or self._disabled.is_set():
            return False
        try:
            terminal = record.get("record_type") == "terminal"
        except Exception:
            self._dropped_record_count += 1
            return False
        if not terminal and self._queue.qsize() >= self._max_queue_size - 1:
            self._dropped_record_count += 1
            return False
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_record_count += 1
            return False
        return True

    def close(self) -> bool:
        """Request a bounded drain and report whether the daemon has exited."""
        self._close_requested.set()
        self._thread.join(self._close_timeout_s)
        joined = not self._thread.is_alive()
        if not joined:
            LOGGER.error(
                "audit writer drain timed out after %.3fs at %s",
                self._close_timeout_s,
                self._path,
            )
        self._report_drops()
        return joined

    def _run(self) -> None:
        handle: Optional[TextIO] = None
        try:
            handle = self._open_file(self._path)
            while not self._close_requested.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                try:
                    line = json.dumps(record, separators=(",", ":")) + "\n"
                    handle.write(line)
                except Exception as exc:
                    self._disable("serialize/write", exc)
                    self._discard_pending()
                    return
                finally:
                    self._queue.task_done()
            try:
                handle.flush()
            except Exception as exc:
                self._disable("flush", exc)
        except Exception as exc:
            self._disable("open", exc)
            self._discard_pending()
        finally:
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:
                    self._disable("close", exc)

    def _default_open(self, path: Path) -> TextIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", buffering=1)

    def _disable(self, operation: str, exc: Exception) -> None:
        self._disabled.set()
        LOGGER.error(
            "audit writer disabled after %s failure at %s: %s",
            operation,
            self._path,
            exc,
        )

    def _discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()
                self._dropped_record_count += 1

    def _report_drops(self) -> None:
        if self._drop_reported or self._dropped_record_count == 0:
            return
        self._drop_reported = True
        LOGGER.warning(
            "audit writer dropped %d record(s) at %s",
            self._dropped_record_count,
            self._path,
        )
