"""Low-rate local control Interface for the V3-A resident fixed cycle."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import threading
from typing import Any, Mapping, Sequence, TextIO

from .resident_control import (
    _absolute_socket_path,
    _receive_json,
    _send_json,
)


SCHEMA_VERSION = "resident_fixed_cycle_control.v1"
_COMMANDS = frozenset({"status", "start", "heartbeat", "cancel"})
_REQUEST_FIELDS = frozenset({"schema_version", "command"})
_START_FIELDS = frozenset(
    {
        "schema_version",
        "command",
        "run_id",
        "requested_cycles",
        "first_dig_point_id",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "run_id",
        "stage",
        "requested_cycles",
        "completed_cycles",
        "current_dig_point_id",
        "terminal",
        "outcome",
        "reason_code",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class ResidentFixedCycleControlServer:
    """Serve one strict request per local Unix connection."""

    def __init__(
        self,
        runtime: Any,
        *,
        socket_path: str,
        request_timeout_s: float = 1.0,
    ) -> None:
        self._runtime = runtime
        self._socket_path = _absolute_socket_path(socket_path)
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, (int, float))
            or not 0.0 < float(request_timeout_s) <= 30.0
        ):
            raise ValueError("request_timeout_s must be within (0, 30]")
        self._request_timeout_s = float(request_timeout_s)
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._lifecycle_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("resident fixed cycle control is already running")
            self._remove_stale_socket()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(self._socket_path)
                path_status = os.lstat(self._socket_path)
                self._socket_identity = (path_status.st_dev, path_status.st_ino)
                os.chmod(self._socket_path, 0o600)
                listener.listen()
                listener.settimeout(0.1)
            except Exception:
                listener.close()
                self._remove_owned_socket()
                raise
            stop = threading.Event()
            self._listener = listener
            self._stop = stop
            self._thread = threading.Thread(
                target=self._serve,
                args=(listener, stop),
                name="resident-fixed-cycle-control",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        with self._lifecycle_lock:
            stop = self._stop
            listener = self._listener
            thread = self._thread
            self._stop = None
            self._listener = None
            self._thread = None
            if stop is not None:
                stop.set()
            if listener is not None:
                listener.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._remove_owned_socket()

    def _serve(self, listener: socket.socket, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if stop.is_set():
                    return
                continue
            with connection:
                connection.settimeout(self._request_timeout_s)
                self._serve_one(connection)

    def _serve_one(self, connection: socket.socket) -> None:
        command: str | None = None
        try:
            command, arguments = _validate_request(_receive_json(connection))
        except (ValueError, OSError):
            response = _error_response(None, "invalid_request", "invalid request")
        else:
            try:
                with self._dispatch_lock:
                    response = self._execute(command, arguments)
            except (ValueError, RuntimeError, OSError):
                response = _error_response(
                    command,
                    "command_failed",
                    f"{command} was rejected by the resident fixed cycle",
                )
            except Exception:
                response = _error_response(
                    command,
                    "internal_error",
                    "resident fixed cycle control failed internally",
                )
        try:
            _send_json(connection, response)
        except (OSError, ValueError, socket.timeout):
            return

    def _execute(self, command: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if command == "start":
            self._runtime.start(**arguments)
        elif command == "heartbeat":
            self._runtime.heartbeat()
        elif command == "cancel":
            self._runtime.cancel()
        elif command != "status":
            raise ValueError("unsupported command")
        return _success_response(command, _status(self._runtime.snapshot))

    def _remove_stale_socket(self) -> None:
        try:
            path_status = os.lstat(self._socket_path)
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_status.st_mode):
            raise FileExistsError("fixed cycle socket path already exists")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(self._socket_path)
        except OSError as exc:
            if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                raise RuntimeError("cannot verify existing fixed cycle socket") from exc
        else:
            raise RuntimeError("fixed cycle socket is already in use")
        finally:
            probe.close()
        current = os.lstat(self._socket_path)
        if (
            (current.st_dev, current.st_ino)
            != (path_status.st_dev, path_status.st_ino)
            or not stat.S_ISSOCK(current.st_mode)
        ):
            raise RuntimeError("fixed cycle socket changed during stale cleanup")
        os.unlink(self._socket_path)

    def _remove_owned_socket(self) -> None:
        identity = self._socket_identity
        if identity is None:
            return
        try:
            current = os.lstat(self._socket_path)
        except FileNotFoundError:
            self._socket_identity = None
            return
        if stat.S_ISSOCK(current.st_mode) and (
            current.st_dev,
            current.st_ino,
        ) == identity:
            os.unlink(self._socket_path)
        self._socket_identity = None


def request_resident_fixed_cycle_control(
    socket_path: str,
    command: str,
    *,
    run_id: str | None = None,
    requested_cycles: int | None = None,
    first_dig_point_id: str | None = None,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    path = _absolute_socket_path(socket_path)
    request = _request(
        command,
        run_id=run_id,
        requested_cycles=requested_cycles,
        first_dig_point_id=first_dig_point_id,
    )
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not 0.0 < float(timeout_s) <= 30.0
    ):
        raise ValueError("timeout_s must be within (0, 30]")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(float(timeout_s))
        connection.connect(path)
        _send_json(connection, request)
        return _validate_response(_receive_json(connection))


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m edge_runtime.resident_fixed_cycle_control"
    )
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("--run-id")
    parser.add_argument("--cycles", type=int, dest="requested_cycles")
    parser.add_argument("--first-dig-point-id")
    arguments = parser.parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        response = request_resident_fixed_cycle_control(
            arguments.socket_path,
            arguments.command,
            run_id=arguments.run_id,
            requested_cycles=arguments.requested_cycles,
            first_dig_point_id=arguments.first_dig_point_id,
        )
    except (OSError, ValueError):
        print("resident fixed cycle control request failed", file=errors)
        return 2
    if not response["ok"]:
        print(
            f"{arguments.command} failed: {response['error']['code']}",
            file=errors,
        )
        return 2
    print(
        json.dumps(
            response,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=output,
    )
    return 0


def _request(
    command: Any,
    *,
    run_id: Any,
    requested_cycles: Any,
    first_dig_point_id: Any,
) -> dict[str, Any]:
    if command not in _COMMANDS:
        raise ValueError("unsupported resident fixed cycle command")
    supplied = any(
        value is not None
        for value in (run_id, requested_cycles, first_dig_point_id)
    )
    if command != "start":
        if supplied:
            raise ValueError("run fields are only valid for start")
        return {"schema_version": SCHEMA_VERSION, "command": command}
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "start",
        "run_id": _identifier("run_id", run_id),
        "requested_cycles": _cycles(requested_cycles),
        "first_dig_point_id": (
            None
            if first_dig_point_id is None
            else _identifier("first_dig_point_id", first_dig_point_id)
        ),
    }


def _validate_request(value: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("invalid fixed cycle control request")
    command = value.get("command")
    expected = _START_FIELDS if command == "start" else _REQUEST_FIELDS
    if command not in _COMMANDS or set(value) != expected:
        raise ValueError("invalid fixed cycle control request fields")
    if command != "start":
        return command, {}
    first = value["first_dig_point_id"]
    return command, {
        "run_id": _identifier("run_id", value["run_id"]),
        "requested_cycles": _cycles(value["requested_cycles"]),
        "first_dig_point_id": (
            None if first is None else _identifier("first_dig_point_id", first)
        ),
    }


def _status(snapshot: Any) -> dict[str, Any]:
    value = {
        name: getattr(snapshot, name)
        for name in _STATUS_FIELDS
    }
    if set(value) != _STATUS_FIELDS:
        raise ValueError("invalid fixed cycle status")
    for name in (
        "run_id",
        "stage",
        "current_dig_point_id",
        "outcome",
        "reason_code",
    ):
        if not isinstance(value[name], str):
            raise ValueError(f"{name} must be a string")
    value["requested_cycles"] = _count(
        "requested_cycles", value["requested_cycles"], maximum=9
    )
    value["completed_cycles"] = _count(
        "completed_cycles", value["completed_cycles"], maximum=9
    )
    if not isinstance(value["terminal"], bool):
        raise ValueError("terminal must be boolean")
    return value


def _success_response(command: str, status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": command,
        "status": dict(status),
        "error": None,
    }


def _error_response(
    command: str | None,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "status": None,
        "error": {"code": code, "message": message},
    }


def _validate_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "ok",
        "command",
        "status",
        "error",
    }:
        raise ValueError("invalid fixed cycle control response")
    if value["schema_version"] != SCHEMA_VERSION or not isinstance(value["ok"], bool):
        raise ValueError("invalid fixed cycle control response schema")
    command = value["command"]
    if command is not None and command not in _COMMANDS:
        raise ValueError("invalid fixed cycle response command")
    if value["ok"]:
        if value["error"] is not None or not isinstance(value["status"], Mapping):
            raise ValueError("invalid successful fixed cycle response")
        status = _status(type("Snapshot", (), dict(value["status"]))())
        return {**value, "status": status}
    error = value["error"]
    if value["status"] is not None or not isinstance(error, Mapping) or set(error) != {
        "code",
        "message",
    }:
        raise ValueError("invalid failed fixed cycle response")
    if not all(isinstance(error[name], str) for name in ("code", "message")):
        raise ValueError("invalid fixed cycle response error")
    return dict(value)


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe identifier")
    return value


def _cycles(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 9:
        raise ValueError("requested_cycles must be within [1, 9]")
    return value


def _count(name: str, value: Any, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} is invalid")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
