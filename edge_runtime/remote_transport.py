"""Length-prefixed TCP JSON transport and concurrent remote behavior server."""

from __future__ import annotations

import json
import logging
import queue
import select
import socket
import threading
from typing import Any, Mapping, Optional


SCHEMA_VERSION = "orin_behavior_rpc.v1"
MAX_FRAME_BYTES = 1024 * 1024
LOGGER = logging.getLogger("orin_edge_remote")
_DEFAULT_PENDING_EVENTS = 256


def send_message(sock: Any, message: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(message),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("JSON frame exceeds maximum size")
    sock.sendall(len(payload).to_bytes(4, "big") + payload)


def receive_message(sock: Any) -> Optional[dict[str, Any]]:
    prefix = _receive_exact(sock, 4)
    if prefix is None:
        return None
    size = int.from_bytes(prefix, "big")
    if size <= 0:
        raise ValueError("JSON frame length must be positive")
    if size > MAX_FRAME_BYTES:
        raise ValueError("JSON frame exceeds maximum size")
    payload = _receive_exact(sock, size)
    if payload is None:
        raise ValueError("connection ended during JSON frame")
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frame must contain UTF-8 JSON") from exc
    if not isinstance(message, dict):
        raise ValueError("JSON frame must contain an object")
    return message


def request_identity(
    request: Mapping[str, Any],
    *,
    expected_type: Optional[str] = None,
) -> tuple[str, int, str]:
    if not isinstance(request, Mapping):
        raise ValueError("request must be an object")
    if request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("request schema_version is invalid")
    if expected_type is not None and request.get("type") != expected_type:
        raise ValueError("request type is invalid")
    session_id = _text("session_id", request.get("session_id"))
    request_id = _text("request_id", request.get("request_id"))
    sequence = request.get("seq")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or sequence > 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("request seq must be an unsigned integer")
    return session_id, sequence, request_id


class ConnectionEventStream:
    """Serialize events on a writer thread without blocking the control loop."""

    def __init__(
        self,
        connection: Any,
        *,
        connection_id: str = "unassigned",
        max_pending_events: int = _DEFAULT_PENDING_EVENTS,
    ) -> None:
        if (
            isinstance(max_pending_events, bool)
            or not isinstance(max_pending_events, int)
            or max_pending_events < 1
        ):
            raise ValueError("max_pending_events must be a positive integer")
        if not isinstance(connection_id, str) or not connection_id.strip():
            raise ValueError("connection_id must be non-empty")
        self._connection = connection
        self._connection_id = connection_id
        self._pending: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(
            maxsize=max_pending_events
        )
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._failure_reason = ""
        self._next_sequence = 0
        self._sent_count = 0
        self._last_sent_type = "none"
        self._last_sent_sequence = -1
        self._thread = threading.Thread(
            target=self._run,
            name="orin-remote-behavior-writer",
            daemon=False,
        )
        self._started = False

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def failure_reason(self) -> str:
        with self._lock:
            return self._failure_reason

    @property
    def sent_summary(self) -> tuple[int, str, int]:
        with self._lock:
            return (
                self._sent_count,
                self._last_sent_type,
                self._last_sent_sequence,
            )

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("ConnectionEventStream has already been started")
        self._started = True
        self._thread.start()

    def emit(self, event: Mapping[str, Any]) -> None:
        should_abort = False
        with self._lock:
            if self._stop.is_set():
                return
            sequenced = {**event, "seq": self._next_sequence}
            try:
                self._pending.put_nowait(sequenced)
            except queue.Full:
                self._failed.set()
                self._stop.set()
                self._failure_reason = "backpressure"
                should_abort = True
            else:
                self._next_sequence += 1
        if should_abort:
            LOGGER.error(
                "remote behavior event writer failed: "
                "connection_id=%s reason=backpressure event_type=%s "
                "event_seq=%s pending_events=%d",
                self._connection_id,
                sequenced.get("type", "none"),
                sequenced.get("seq", -1),
                self._pending.qsize(),
            )
            self._shutdown_connection()

    def close(self) -> None:
        self._stop.set()
        if not self._started:
            return
        self._thread.join(timeout=0.5)
        if self._thread.is_alive():
            self._shutdown_connection()
            self._thread.join(timeout=0.5)
        if self._thread.is_alive():
            raise RuntimeError("remote behavior event writer did not stop")

    def _run(self) -> None:
        while not self._stop.is_set() or not self._pending.empty():
            try:
                event = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if event is not None:
                    send_message(self._connection, event)
                    with self._lock:
                        self._sent_count += 1
                        self._last_sent_type = str(event.get("type", "unknown"))
                        self._last_sent_sequence = int(event.get("seq", -1))
            except (OSError, ValueError) as exc:
                self._failed.set()
                self._stop.set()
                with self._lock:
                    self._failure_reason = "send_error"
                LOGGER.error(
                    "remote behavior event writer failed: "
                    "connection_id=%s reason=send_error event_type=%s "
                    "event_seq=%s pending_events=%d error=%s: %s",
                    self._connection_id,
                    event.get("type") if event is not None else "none",
                    event.get("seq") if event is not None else -1,
                    self._pending.qsize(),
                    type(exc).__name__,
                    exc,
                )
                self._shutdown_connection()
                return
            finally:
                self._pending.task_done()

    def _shutdown_connection(self) -> None:
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


class RemoteBehaviorServer:
    """Serve status and a globally serialized Follow over concurrent clients."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        allowed_client_host: str,
        executor: Any,
        status_interval_s: float = 0.2,
    ) -> None:
        if not isinstance(bind_port, int) or isinstance(bind_port, bool):
            raise ValueError("remote behavior bind_port must be an integer")
        if bind_port < 0 or bind_port > 65535:
            raise ValueError("remote behavior bind_port is out of range")
        self._bind_host = _text("remote behavior bind_host", bind_host)
        self._bind_port = bind_port
        self._allowed_client_host = socket.gethostbyname(
            _text("remote behavior allowed_client_host", allowed_client_host)
        )
        self._executor = executor
        if status_interval_s <= 0.0:
            raise ValueError("remote behavior status_interval_s must be positive")
        self._status_interval_s = float(status_interval_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="orin-remote-behavior-server",
            daemon=False,
        )
        self._listener: Optional[socket.socket] = None
        self._socket_lock = threading.Lock()
        self._client_threads: set[threading.Thread] = set()
        self._connections: set[socket.socket] = set()
        self._next_connection_sequence = 1
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RemoteBehaviorServer has already been started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self._bind_host, self._bind_port))
            listener.listen(2)
            listener.settimeout(0.2)
        except Exception:
            listener.close()
            raise
        with self._socket_lock:
            self._listener = listener
        self._started = True
        self._thread.start()

    @property
    def bound_port(self) -> int:
        with self._socket_lock:
            if self._listener is None:
                raise RuntimeError("RemoteBehaviorServer is not started")
            return int(self._listener.getsockname()[1])

    def close(self) -> None:
        self._stop.set()
        with self._socket_lock:
            connections = tuple(self._connections)
            listener = self._listener
            client_threads = tuple(self._client_threads)
        for sock in (*connections, listener):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if self._started:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("RemoteBehaviorServer did not stop")
            for thread in client_threads:
                thread.join(timeout=1.0)
                if thread.is_alive():
                    raise RuntimeError("remote behavior client did not stop")
        self._executor.close(emit_result=False)

    def serve_connection(
        self,
        connection: socket.socket,
        *,
        connection_id: str = "rpc-direct",
        peer: str = "unknown",
    ) -> None:
        LOGGER.info(
            "remote behavior connection opened: connection_id=%s peer=%s",
            connection_id,
            peer,
        )
        stream = ConnectionEventStream(
            connection,
            connection_id=connection_id,
        )
        stream.start()
        emit = stream.emit
        close_reason = "server_shutdown"

        try:
            connection.settimeout(None)
            emit(self._executor.status_event())
            while not self._stop.is_set() and not stream.failed:
                readable, _, _ = select.select(
                    [connection],
                    [],
                    [],
                    self._status_interval_s,
                )
                if not readable:
                    self._executor.watchdog()
                    emit(self._executor.status_event())
                    continue
                request = receive_message(connection)
                if request is None:
                    close_reason = "peer_eof"
                    break
                LOGGER.info(
                    "remote behavior request received: connection_id=%s "
                    "peer=%s type=%s session_id=%s request_id=%s "
                    "request_seq=%s",
                    connection_id,
                    peer,
                    request.get("type", "unknown"),
                    request.get("session_id", "unknown"),
                    request.get("request_id", "unknown"),
                    request.get("seq", -1),
                )
                self._executor.handle(request, emit)
            if stream.failed:
                close_reason = stream.failure_reason or "writer_failed"
        except (OSError, ValueError) as exc:
            close_reason = "receive_error"
            LOGGER.warning(
                "remote behavior connection receive failed: "
                "connection_id=%s peer=%s reason=%s error=%s: %s",
                connection_id,
                peer,
                close_reason,
                type(exc).__name__,
                exc,
            )
        finally:
            self._executor.disconnect(emit)
            stream.close()
            if stream.failed:
                close_reason = stream.failure_reason or "writer_failed"
            sent_count, last_event_type, last_event_sequence = stream.sent_summary
            LOGGER.info(
                "remote behavior connection closed: connection_id=%s peer=%s "
                "reason=%s events_sent=%d last_event_type=%s "
                "last_event_seq=%d pending_events=%d",
                connection_id,
                peer,
                close_reason,
                sent_count,
                last_event_type,
                last_event_sequence,
                stream.pending_count,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._socket_lock:
                listener = self._listener
            if listener is None:
                return
            try:
                connection, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if address[0] != self._allowed_client_host:
                LOGGER.warning(
                    "reject remote behavior client %s; expected %s",
                    address[0],
                    self._allowed_client_host,
                )
                connection.close()
                continue
            connection_id = f"rpc-{self._next_connection_sequence:06d}"
            self._next_connection_sequence += 1
            peer = f"{address[0]}:{address[1]}"
            thread = threading.Thread(
                target=self._serve_client,
                args=(connection, connection_id, peer),
                name="orin-remote-behavior-client",
                daemon=False,
            )
            with self._socket_lock:
                self._connections.add(connection)
                self._client_threads.add(thread)
            thread.start()

    def _serve_client(
        self,
        connection: socket.socket,
        connection_id: str,
        peer: str,
    ) -> None:
        try:
            self.serve_connection(
                connection,
                connection_id=connection_id,
                peer=peer,
            )
        finally:
            connection.close()
            with self._socket_lock:
                self._connections.discard(connection)
                self._client_threads.discard(threading.current_thread())


def _receive_exact(sock: Any, size: int) -> Optional[bytes]:
    received = bytearray()
    while len(received) < size:
        chunk = sock.recv(size - len(received))
        if not chunk:
            if not received:
                return None
            raise ValueError("connection ended during frame")
        received.extend(chunk)
    return bytes(received)


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be non-empty" % name)
    return value
