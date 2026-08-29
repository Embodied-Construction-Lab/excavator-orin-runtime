"""Bounded local data link between the resident owner and one ACT worker.

The owner is the sole STM32 authority.  This module only transports strict
``resident_act_state.v1`` frames to a local policy worker and strict
``resident_policy_candidate.v2`` frames back to an injected callback.  It has
no serial dependency and a disconnect therefore cannot alter serial ownership.

There may be only one active worker.  A second connection is accepted and
immediately closed while the first remains active; it never replaces the
current worker.  Malformed, oversized, or callback-exception inbound data closes
only that worker connection (fail closed) while leaving the listener ready for
a clean reconnect.  A callback that returns a policy-level passive rejection
(for example, an old activation generation) keeps the transport connected.
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
import select
import socket
import stat
import threading
from typing import Callable

from .resident_protocol import MAX_CANDIDATE_BYTES, decode_motion_candidate
from .resident_state import MAX_RESIDENT_STATE_BYTES, decode_resident_state


MAX_FRAME_BYTES = 4096
_HEADER_BYTES = 4
_SELECT_TIMEOUT_S = 0.02
_UNIX_PATH_MAX_BYTES = 107


class ResidentActDataLink:
    """One-listener, one-worker, latest-state-only Unix stream transport."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        on_candidate: Callable[[bytes], object],
        on_connection_lost: Callable[[], object] | None = None,
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("resident ACT socket path must be absolute")
        if len(os.fsencode(path)) > _UNIX_PATH_MAX_BYTES:
            raise ValueError("resident ACT socket path exceeds the Unix limit")
        if not callable(on_candidate):
            raise ValueError("on_candidate must be a callback")
        if on_connection_lost is not None and not callable(on_connection_lost):
            raise ValueError("on_connection_lost must be a callback")
        if MAX_RESIDENT_STATE_BYTES != MAX_FRAME_BYTES:
            raise RuntimeError("resident state and data-link frame limits disagree")
        if MAX_CANDIDATE_BYTES != MAX_FRAME_BYTES:
            raise RuntimeError("resident candidate and data-link frame limits disagree")

        self._path = path
        self._on_candidate = on_candidate
        self._on_connection_lost = on_connection_lost
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._active_connection: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._connection_threads: set[threading.Thread] = set()
        self._ready = False
        self._closed = False
        self._owned_socket_identity: tuple[int, int] | None = None
        self._latest_state: bytes | None = None
        self._state_revision = 0
        self._rejected_connection_count = 0
        self._logger = logging.getLogger("resident_act_data_link")

    @property
    def socket_path(self) -> Path:
        return self._path

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._active_connection is not None

    @property
    def rejected_connection_count(self) -> int:
        with self._lock:
            return self._rejected_connection_count

    def start(self) -> None:
        """Bind the local socket and start accepting one ACT worker."""

        with self._lock:
            if self._closed:
                raise RuntimeError("resident ACT data link is closed")
            if self._ready:
                return

            self._prepare_parent()
            self._remove_stale_socket()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(os.fspath(self._path))
                path_stat = os.lstat(self._path)
                self._owned_socket_identity = (path_stat.st_dev, path_stat.st_ino)
                os.chmod(self._path, 0o600)
                listener.listen(4)
                listener.settimeout(0.1)
            except BaseException:
                listener.close()
                self._unlink_owned_socket_if_present()
                raise

            self._listener = listener
            self._ready = True
            accept_thread = threading.Thread(
                target=self._accept_loop,
                name="resident-act-data-link-accept",
                daemon=True,
            )
            self._accept_thread = accept_thread
            accept_thread.start()

    def publish(self, payload: bytes) -> None:
        """Publish one strict state frame, replacing any unsent older state."""

        decode_resident_state(payload)
        with self._lock:
            if not self._ready or self._closed:
                raise RuntimeError("resident ACT data link is not ready")
            self._latest_state = payload
            self._state_revision += 1

    def close(self) -> None:
        """Stop the data link and remove only the socket inode it created."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._ready = False
            self._stop.set()
            listener = self._listener
            self._listener = None
            active = self._active_connection
            self._active_connection = None
            accept_thread = self._accept_thread
            connection_threads = tuple(self._connection_threads)

        _close_socket(listener)
        _close_socket(active)
        current = threading.current_thread()
        if accept_thread is not None and accept_thread is not current:
            accept_thread.join(timeout=1.0)
        for connection_thread in connection_threads:
            if connection_thread is not current:
                connection_thread.join(timeout=1.0)
        self._unlink_owned_socket_if_present()

    def _prepare_parent(self) -> None:
        parent = self._path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_stat = parent.stat()
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RuntimeError("resident ACT socket parent is not a directory")

    def _remove_stale_socket(self) -> None:
        try:
            path_stat = os.lstat(self._path)
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_stat.st_mode):
            raise RuntimeError("resident ACT socket path already exists and is not a socket")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(os.fspath(self._path))
        except OSError as exc:
            if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                raise RuntimeError("cannot verify the existing resident ACT socket") from exc
        else:
            raise RuntimeError("resident ACT socket is already in use")
        finally:
            probe.close()

        try:
            current = os.lstat(self._path)
        except FileNotFoundError:
            return
        if (
            current.st_dev != path_stat.st_dev
            or current.st_ino != path_stat.st_ino
            or not stat.S_ISSOCK(current.st_mode)
        ):
            raise RuntimeError("resident ACT socket changed during stale cleanup")
        os.unlink(self._path)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                listener = self._listener
            if listener is None:
                break
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue

            connection.setblocking(False)
            with self._lock:
                if self._closed or self._active_connection is not None:
                    self._rejected_connection_count += 1
                    rejected = True
                else:
                    self._active_connection = connection
                    rejected = False
                    connection_thread = threading.Thread(
                        target=self._serve_connection,
                        args=(connection,),
                        name="resident-act-data-link-worker",
                        daemon=True,
                    )
                    self._connection_threads.add(connection_thread)
            if rejected:
                _close_socket(connection)
                continue
            connection_thread.start()

    def _serve_connection(self, connection: socket.socket) -> None:
        receive_buffer = bytearray()
        expected_payload_bytes: int | None = None
        outbound = b""
        outbound_offset = 0
        outbound_revision = 0
        last_sent_revision = 0
        try:
            while not self._stop.is_set():
                if not outbound:
                    outbound, outbound_revision = self._next_state_frame(
                        last_sent_revision
                    )
                    outbound_offset = 0
                elif outbound_offset == 0:
                    replacement, replacement_revision = self._next_state_frame(
                        outbound_revision
                    )
                    if replacement:
                        outbound = replacement
                        outbound_revision = replacement_revision

                readable, writable, _ = select.select(
                    [connection],
                    [connection] if outbound else [],
                    [],
                    _SELECT_TIMEOUT_S,
                )
                if readable:
                    chunk = connection.recv(MAX_FRAME_BYTES + _HEADER_BYTES)
                    if not chunk:
                        break
                    receive_buffer.extend(chunk)
                    expected_payload_bytes = self._consume_inbound_frames(
                        receive_buffer,
                        expected_payload_bytes,
                    )
                if writable and outbound:
                    sent = connection.send(outbound[outbound_offset:])
                    if sent <= 0:
                        break
                    outbound_offset += sent
                    if outbound_offset == len(outbound):
                        last_sent_revision = outbound_revision
                        outbound = b""
                        outbound_offset = 0
        except (OSError, ValueError, RuntimeError) as exc:
            if not self._stop.is_set():
                self._logger.warning("ACT worker data link closed: %s", exc)
        finally:
            _close_socket(connection)
            with self._lock:
                notify_connection_lost = (
                    self._active_connection is connection
                    and not self._stop.is_set()
                )
                if self._active_connection is connection:
                    self._active_connection = None
                self._connection_threads.discard(threading.current_thread())
            if notify_connection_lost and self._on_connection_lost is not None:
                try:
                    self._on_connection_lost()
                except Exception as exc:
                    self._logger.error(
                        "ACT worker disconnect callback failed: %s",
                        exc,
                    )

    def _next_state_frame(self, after_revision: int) -> tuple[bytes, int]:
        with self._lock:
            if (
                self._latest_state is None
                or self._state_revision <= after_revision
            ):
                return b"", after_revision
            payload = self._latest_state
            revision = self._state_revision
        return len(payload).to_bytes(_HEADER_BYTES, "big") + payload, revision

    def _consume_inbound_frames(
        self,
        receive_buffer: bytearray,
        expected_payload_bytes: int | None,
    ) -> int | None:
        while True:
            if expected_payload_bytes is None:
                if len(receive_buffer) < _HEADER_BYTES:
                    return None
                expected_payload_bytes = int.from_bytes(
                    receive_buffer[:_HEADER_BYTES],
                    "big",
                )
                del receive_buffer[:_HEADER_BYTES]
                if not 0 < expected_payload_bytes <= MAX_FRAME_BYTES:
                    raise ValueError("ACT worker frame length is invalid")
            if len(receive_buffer) < expected_payload_bytes:
                return expected_payload_bytes
            payload = bytes(receive_buffer[:expected_payload_bytes])
            del receive_buffer[:expected_payload_bytes]
            expected_payload_bytes = None
            decode_motion_candidate(payload)
            try:
                self._on_candidate(payload)
            except Exception as exc:
                raise RuntimeError("ACT candidate callback rejected the frame") from exc

    def _unlink_owned_socket_if_present(self) -> None:
        with self._lock:
            identity = self._owned_socket_identity
            self._owned_socket_identity = None
        if identity is None:
            return
        try:
            path_stat = os.lstat(self._path)
        except FileNotFoundError:
            return
        if (
            stat.S_ISSOCK(path_stat.st_mode)
            and (path_stat.st_dev, path_stat.st_ino) == identity
        ):
            os.unlink(self._path)


def _close_socket(connection: socket.socket | None) -> None:
    if connection is None:
        return
    try:
        connection.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        connection.close()
    except OSError:
        pass
