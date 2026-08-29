"""Best-effort asynchronous evidence for the resident STM32 command seam."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
import time
from typing import Callable, ContextManager, Protocol, TextIO


ACTION_AUDIT_SCHEMA_VERSION = "resident_action_audit.v1"


class ResidentActionAuditSink(Protocol):
    """Non-blocking evidence seam used by the motion owner."""

    def emit(self, event_type: str, **fields: object) -> bool: ...


def emit_action_audit(
    sink: ResidentActionAuditSink | None,
    event_type: str,
    **fields: object,
) -> bool:
    """Contain every evidence adapter failure outside the motion path."""

    if sink is None:
        return False
    try:
        return sink.emit(event_type, **fields) is True
    except Exception:
        return False


def _open_append(path: Path) -> ContextManager[TextIO]:
    return path.open("a", encoding="utf-8")


@dataclass(frozen=True)
class ResidentActionAuditStatus:
    queued_event_count: int
    written_event_count: int
    dropped_event_count: int
    error_count: int
    closed: bool
    writer_alive: bool


class AsyncResidentActionAudit:
    """Append compact JSONL without performing file I/O in control callers."""

    def __init__(
        self,
        path: Path,
        *,
        queue_capacity: int = 4096,
        wall_time_ns: Callable[[], int] = time.time_ns,
        open_append: Callable[[Path], ContextManager[TextIO]] = _open_append,
    ) -> None:
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
            raise ValueError("queue_capacity must be an integer")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self._path = Path(path)
        self._wall_time_ns = wall_time_ns
        self._open_append = open_append
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.Lock()
        self._queued_event_count = 0
        self._written_event_count = 0
        self._dropped_event_count = 0
        self._error_count = 0
        self._closed = False
        self._closing = threading.Event()
        self._writer_failed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="resident-action-audit",
            daemon=True,
        )
        self._thread.start()

    def emit(self, event_type: str, **fields: object) -> bool:
        """Queue one immutable record immediately; never wait for the writer."""

        try:
            record = {
                "schema_version": ACTION_AUDIT_SCHEMA_VERSION,
                "event_type": event_type,
                "wall_time_ns": self._wall_time_ns(),
                **fields,
            }
            with self._lock:
                if self._closed or self._writer_failed.is_set():
                    self._dropped_event_count += 1
                    return False
                self._queue.put_nowait(record)
                self._queued_event_count += 1
            return True
        except Exception:
            with self._lock:
                self._dropped_event_count += 1
                self._error_count += 1
            return False

    def status(self) -> ResidentActionAuditStatus:
        with self._lock:
            return ResidentActionAuditStatus(
                queued_event_count=self._queued_event_count,
                written_event_count=self._written_event_count,
                dropped_event_count=self._dropped_event_count,
                error_count=self._error_count,
                closed=self._closed,
                writer_alive=self._thread.is_alive(),
            )

    def close(self, *, timeout_s: float = 1.0) -> None:
        """Request bounded shutdown; evidence cleanup never raises to the owner."""

        try:
            with self._lock:
                self._closed = True
                self._closing.set()
            self._thread.join(timeout=max(0.0, float(timeout_s)))
        except Exception:
            with self._lock:
                self._error_count += 1

    def _run(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._open_append(self._path) as handle:
                while not (self._closing.is_set() and self._queue.empty()):
                    try:
                        item = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        handle.write(
                            json.dumps(
                                item,
                                ensure_ascii=True,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        handle.flush()
                        with self._lock:
                            self._written_event_count += 1
                    except Exception:
                        with self._lock:
                            self._error_count += 1
        except Exception:
            self._writer_failed.set()
            with self._lock:
                self._error_count += 1
