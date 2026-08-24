"""Local control plane for the resident Orin motion core.

This module deliberately carries no state samples and no policy candidates.
It exposes only the four low-rate control operations needed by a Mission
orchestrator, over a local filesystem Unix socket.  The resident motion core
remains the sole authority for every state transition.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
import threading
from typing import Any, Callable, Mapping, Sequence, TextIO

from .resident_motion import ControlMode, HandoffPhase, PolicyBinding


SCHEMA_VERSION = "resident_motion_control.v1"
MAX_CONTROL_FRAME_BYTES = 4096
_MAX_UNIX_PATH_BYTES = 107
_COMMANDS = frozenset(
    {"status", "renew_lease", "activate_rl", "activate_act", "terminal_disarm"}
)
DEFAULT_MISSION_LEASE_MS = 1500
_REQUEST_FIELDS = frozenset({"schema_version", "command"})
_ACTIVATE_ACT_REQUEST_FIELDS = frozenset(
    {"schema_version", "command", "max_steps"}
)
_RESPONSE_FIELDS = frozenset(
    {"schema_version", "ok", "command", "status", "error"}
)
_STATUS_FIELDS = frozenset(
    {
        "phase",
        "control_generation",
        "active",
        "target",
        "last_handoff_latency_ms",
        "rl_is_active",
        "act_is_active",
        "act_worker_ready",
        "act_segment_generation",
        "act_segment_max_steps",
        "act_segment_completed_steps",
        "act_segment_complete",
        "mission_lease_active",
        "is_operational",
    }
)
_BINDING_FIELDS = frozenset({"source", "mode"})
_ERROR_FIELDS = frozenset({"code", "message"})
_ERROR_CODES = frozenset({"invalid_request", "command_failed", "internal_error"})


class ResidentMotionControlServer:
    """Sequential, one-request-per-connection Unix control server."""

    def __init__(
        self,
        core: Any,
        *,
        socket_path: str,
        request_timeout_s: float = 1.0,
        act_worker_ready: Callable[[], bool] | None = None,
        rl_behavior_idle: Callable[[], bool] | None = None,
        activate_act_while_rl_idle: Callable[[Callable[[], Any]], Any]
        | None = None,
        mission_lease_ms: int = DEFAULT_MISSION_LEASE_MS,
    ) -> None:
        self._core = core
        self._socket_path = _absolute_socket_path(socket_path)
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(float(request_timeout_s))
            or float(request_timeout_s) <= 0.0
        ):
            raise ValueError("request_timeout_s must be finite and positive")
        self._request_timeout_s = float(request_timeout_s)
        if act_worker_ready is not None and not callable(act_worker_ready):
            raise ValueError("act_worker_ready must be callable")
        self._act_worker_ready = (
            (lambda: True) if act_worker_ready is None else act_worker_ready
        )
        if rl_behavior_idle is not None and not callable(rl_behavior_idle):
            raise ValueError("rl_behavior_idle must be callable")
        self._rl_behavior_idle = (
            (lambda: True) if rl_behavior_idle is None else rl_behavior_idle
        )
        if activate_act_while_rl_idle is not None and not callable(
            activate_act_while_rl_idle
        ):
            raise ValueError("activate_act_while_rl_idle must be callable")
        self._activate_act_while_rl_idle = activate_act_while_rl_idle
        if (
            isinstance(mission_lease_ms, bool)
            or not isinstance(mission_lease_ms, int)
            or not 500 <= mission_lease_ms <= 5000
        ):
            raise ValueError("mission_lease_ms must be an integer within [500, 5000]")
        self._mission_lease_ms = mission_lease_ms
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
        """Bind the local endpoint and start serving before returning."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("resident motion control server is already running")
            self._remove_stale_socket()

            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(self._socket_path)
                socket_stat = os.lstat(self._socket_path)
                self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
                os.chmod(self._socket_path, 0o600)
                listener.listen()
                listener.settimeout(0.1)
            except Exception:
                listener.close()
                self._remove_owned_socket()
                raise

            self._listener = listener
            stop = threading.Event()
            self._stop = stop
            self._thread = threading.Thread(
                target=self._serve,
                args=(listener, stop),
                name="resident-motion-control",
                daemon=True,
            )
            self._thread.start()

    def _remove_stale_socket(self) -> None:
        try:
            path_stat = os.lstat(self._socket_path)
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_stat.st_mode):
            raise FileExistsError(
                "resident motion control socket path already exists"
            )

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(self._socket_path)
        except OSError as exc:
            if exc.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                raise RuntimeError(
                    "cannot verify the existing resident motion control socket"
                ) from exc
        else:
            raise RuntimeError("resident motion control socket is already in use")
        finally:
            probe.close()

        try:
            current = os.lstat(self._socket_path)
        except FileNotFoundError:
            return
        if (
            current.st_dev != path_stat.st_dev
            or current.st_ino != path_stat.st_ino
            or not stat.S_ISSOCK(current.st_mode)
        ):
            raise RuntimeError(
                "resident motion control socket changed during stale cleanup"
            )
        os.unlink(self._socket_path)

    def close(self) -> None:
        """Stop accepting requests and remove only this server's socket inode.

        Closing the control transport is intentionally not a motion command;
        only the explicit ``terminal_disarm`` request calls the matching core
        method.
        """

        with self._lifecycle_lock:
            stop = self._stop
            if stop is not None:
                stop.set()
            listener = self._listener
            thread = self._thread
            self._listener = None
            self._stop = None
            self._thread = None
            if listener is not None:
                listener.close()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._remove_owned_socket()

    def __enter__(self) -> "ResidentMotionControlServer":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

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
        max_steps: int | None = None
        try:
            request = _receive_json(connection)
            command, max_steps = _validate_request(request)
        except (ValueError, OSError):
            response = _error_response(
                command=None,
                code="invalid_request",
                message="request must be one valid resident_motion_control.v1 frame",
            )
        else:
            try:
                with self._dispatch_lock:
                    response = self._execute(command, max_steps=max_steps)
            except (ValueError, RuntimeError, OSError):
                response = _error_response(
                    command=command,
                    code="command_failed",
                    message=f"{command} was rejected by the motion core",
                )
            except Exception:
                response = _error_response(
                    command=command,
                    code="internal_error",
                    message="resident motion control failed internally",
                )

        try:
            _send_json(connection, response)
        except (OSError, ValueError, socket.timeout):
            return

    def _execute(
        self,
        command: str,
        *,
        max_steps: int | None,
    ) -> dict[str, Any]:
        act_worker_ready = self._act_worker_is_ready()
        if command == "renew_lease":
            self._core.renew_mission_lease(lease_ms=self._mission_lease_ms)
        elif command == "activate_rl":
            self._core.renew_mission_lease(lease_ms=self._mission_lease_ms)
            self._core.activate_rl()
        elif command == "activate_act":
            if not act_worker_ready:
                raise RuntimeError("ACT worker is not ready")
            if max_steps is None:
                raise ValueError("activate_act requires max_steps")

            def activate_act() -> Any:
                self._core.renew_mission_lease(
                    lease_ms=self._mission_lease_ms
                )
                return self._core.activate_act(max_steps=max_steps)

            if self._activate_act_while_rl_idle is None:
                if not self._rl_behavior_is_idle():
                    raise RuntimeError("RL behavior is active or unavailable")
                activate_act()
            else:
                self._activate_act_while_rl_idle(activate_act)
        elif command == "terminal_disarm":
            self._core.terminal_disarm()
        elif command != "status":
            raise ValueError("unsupported resident motion command")
        return _success_response(
            command=command,
            status=_status(
                self._core,
                act_worker_ready=act_worker_ready,
            ),
        )

    def _act_worker_is_ready(self) -> bool:
        try:
            ready = self._act_worker_ready()
        except Exception:
            return False
        return ready if isinstance(ready, bool) else False

    def _rl_behavior_is_idle(self) -> bool:
        try:
            idle = self._rl_behavior_idle()
        except Exception:
            return False
        return idle is True

    def _remove_owned_socket(self) -> None:
        identity = self._socket_identity
        if identity is None:
            return
        try:
            current = os.lstat(self._socket_path)
        except FileNotFoundError:
            self._socket_identity = None
            return
        if (
            (current.st_dev, current.st_ino) == identity
            and stat.S_ISSOCK(current.st_mode)
        ):
            os.unlink(self._socket_path)
        self._socket_identity = None


def request_resident_motion_control(
    socket_path: str,
    command: str,
    *,
    max_steps: int | None = None,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    """Send one control request on one short-lived local connection."""

    path = _absolute_socket_path(socket_path)
    validated_command = _command(command)
    validated_max_steps = _request_max_steps(validated_command, max_steps)
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0.0
    ):
        raise ValueError("timeout_s must be finite and positive")

    request = {
        "schema_version": SCHEMA_VERSION,
        "command": validated_command,
    }
    if validated_max_steps is not None:
        request["max_steps"] = validated_max_steps
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(float(timeout_s))
        connection.connect(path)
        _send_json(connection, request)
        response = _receive_json(connection)
    return _validate_response(response)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the minimal local control client used by Orin commissioning."""

    parser = argparse.ArgumentParser(
        prog="python3 -m edge_runtime.resident_control",
        description="Send one request to the resident Orin motion core.",
    )
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("--max-steps", type=int, default=None)
    arguments = parser.parse_args(argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr

    try:
        response = _validate_response(
            request_resident_motion_control(
                arguments.socket_path,
                arguments.command,
                max_steps=arguments.max_steps,
            )
        )
    except (OSError, ValueError):
        print("resident motion control request failed", file=errors)
        return 2

    if not response["ok"]:
        print(
            f"{arguments.command} failed: {response['error']['code']}",
            file=errors,
        )
        return 2

    encoded = json.dumps(
        response,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    print(encoded, file=output)
    return 0


def _absolute_socket_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("socket_path must be an absolute filesystem path")
    if "\x00" in value or not Path(value).is_absolute():
        raise ValueError("socket_path must be an absolute filesystem path")
    if len(os.fsencode(value)) > _MAX_UNIX_PATH_BYTES:
        raise ValueError("socket_path is too long for a Unix domain socket")
    return value


def _command(value: Any) -> str:
    if not isinstance(value, str) or value not in _COMMANDS:
        raise ValueError("command is not supported by resident_motion_control.v1")
    return value


def _request_max_steps(command: str, value: Any) -> int | None:
    if command != "activate_act":
        if value is not None:
            raise ValueError("max_steps is only valid for activate_act")
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 2000
    ):
        raise ValueError("activate_act max_steps must be within [1, 2000]")
    return value


def _validate_request(value: Any) -> tuple[str, int | None]:
    if not isinstance(value, Mapping):
        raise ValueError("resident motion control request fields are invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("resident motion control schema_version is invalid")
    command = _command(value.get("command"))
    expected_fields = (
        _ACTIVATE_ACT_REQUEST_FIELDS
        if command == "activate_act"
        else _REQUEST_FIELDS
    )
    if set(value) != expected_fields:
        raise ValueError("resident motion control request fields are invalid")
    max_steps = value.get("max_steps")
    return command, _request_max_steps(command, max_steps)


def _status(core: Any, *, act_worker_ready: bool) -> dict[str, Any]:
    atomic_snapshot = getattr(core, "control_status_snapshot", None)
    if callable(atomic_snapshot):
        control_snapshot = atomic_snapshot()
        snapshot = control_snapshot.motion
        act_segment = control_snapshot.act_segment
        rl_is_active = control_snapshot.rl_is_active
        act_is_active = control_snapshot.act_is_active
        mission_lease_active = control_snapshot.mission_lease_active
        is_operational = control_snapshot.is_operational
    else:
        # Compatibility for small injected test doubles.  Production cores
        # always expose the atomic control_status_snapshot boundary.
        snapshot = core.snapshot()
        act_segment = core.act_segment_snapshot()
        rl_is_active = core.rl_is_active
        act_is_active = core.act_is_active
        mission_lease_active = core.mission_lease_is_active
        is_operational = core.is_operational
    phase = HandoffPhase(snapshot.phase)
    generation = _uint64("control_generation", snapshot.generation)
    latency = snapshot.last_handoff_latency_ms
    if latency is not None:
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise ValueError("last_handoff_latency_ms must be finite and nonnegative")
        latency = float(latency)
    segment_max_steps = _optional_segment_max_steps(
        "act_segment_max_steps", act_segment.max_steps
    )
    segment_completed_steps = _uint64(
        "act_segment_completed_steps", act_segment.completed_steps
    )
    if segment_max_steps is not None and segment_completed_steps > segment_max_steps:
        raise ValueError("ACT segment completed_steps exceeds max_steps")
    return {
        "phase": phase.value,
        "control_generation": generation,
        "active": _binding(snapshot.active_binding),
        "target": _binding(snapshot.target_binding),
        "last_handoff_latency_ms": latency,
        "rl_is_active": _boolean("rl_is_active", rl_is_active),
        "act_is_active": _boolean("act_is_active", act_is_active),
        "act_worker_ready": _boolean("act_worker_ready", act_worker_ready),
        "act_segment_generation": _optional_uint64(
            "act_segment_generation", act_segment.generation
        ),
        "act_segment_max_steps": segment_max_steps,
        "act_segment_completed_steps": segment_completed_steps,
        "act_segment_complete": _boolean(
            "act_segment_complete", act_segment.complete
        ),
        "mission_lease_active": _boolean(
            "mission_lease_active", mission_lease_active
        ),
        "is_operational": _boolean("is_operational", is_operational),
    }


def _binding(value: PolicyBinding | None) -> dict[str, str] | None:
    if value is None:
        return None
    binding = PolicyBinding(value.source, value.mode)
    return {"source": binding.source, "mode": binding.mode.value}


def _success_response(*, command: str, status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": command,
        "status": dict(status),
        "error": None,
    }


def _error_response(*, command: str | None, code: str, message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "status": None,
        "error": {"code": code, "message": message},
    }


def _send_json(connection: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not payload or len(payload) > MAX_CONTROL_FRAME_BYTES:
        raise ValueError("resident motion control frame exceeds 4096 bytes")
    connection.sendall(len(payload).to_bytes(4, "big") + payload)


def _receive_json(connection: socket.socket) -> Any:
    header = _receive_exact(connection, 4)
    size = int.from_bytes(header, "big")
    if size <= 0 or size > MAX_CONTROL_FRAME_BYTES:
        raise ValueError("resident motion control frame size is invalid")
    payload = _receive_exact(connection, size)
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_raise_invalid_number(value)),
    )


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise ValueError("resident motion control frame ended early")
        result.extend(chunk)
    return bytes(result)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("resident motion control JSON contains duplicate fields")
        result[key] = value
    return result


def _raise_invalid_number(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")


def _validate_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESPONSE_FIELDS:
        raise ValueError("resident motion control response fields are invalid")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("resident motion control response schema is invalid")
    ok = _boolean("ok", value["ok"])
    command = value["command"]
    if command is not None:
        command = _command(command)

    if ok:
        if command is None or value["error"] is not None:
            raise ValueError("successful resident motion response is inconsistent")
        status = _validate_status(value["status"])
        error = None
    else:
        if value["status"] is not None:
            raise ValueError("failed resident motion response must not contain status")
        error = _validate_error(value["error"])
        status = None
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "command": command,
        "status": status,
        "error": error,
    }


def _validate_status(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STATUS_FIELDS:
        raise ValueError("resident motion status fields are invalid")
    phase = HandoffPhase(value["phase"]).value
    latency = value["last_handoff_latency_ms"]
    if latency is not None:
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise ValueError("last_handoff_latency_ms must be finite and nonnegative")
        latency = float(latency)
    segment_max_steps = _optional_segment_max_steps(
        "act_segment_max_steps", value["act_segment_max_steps"]
    )
    segment_completed_steps = _uint64(
        "act_segment_completed_steps", value["act_segment_completed_steps"]
    )
    if segment_max_steps is not None and segment_completed_steps > segment_max_steps:
        raise ValueError("ACT segment completed_steps exceeds max_steps")
    return {
        "phase": phase,
        "control_generation": _uint64(
            "control_generation", value["control_generation"]
        ),
        "active": _validate_binding(value["active"]),
        "target": _validate_binding(value["target"]),
        "last_handoff_latency_ms": latency,
        "rl_is_active": _boolean("rl_is_active", value["rl_is_active"]),
        "act_is_active": _boolean("act_is_active", value["act_is_active"]),
        "act_worker_ready": _boolean(
            "act_worker_ready", value["act_worker_ready"]
        ),
        "act_segment_generation": _optional_uint64(
            "act_segment_generation", value["act_segment_generation"]
        ),
        "act_segment_max_steps": segment_max_steps,
        "act_segment_completed_steps": segment_completed_steps,
        "act_segment_complete": _boolean(
            "act_segment_complete", value["act_segment_complete"]
        ),
        "mission_lease_active": _boolean(
            "mission_lease_active", value["mission_lease_active"]
        ),
        "is_operational": _boolean("is_operational", value["is_operational"]),
    }


def _validate_binding(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _BINDING_FIELDS:
        raise ValueError("resident motion binding fields are invalid")
    source = value["source"]
    if not isinstance(source, str) or not source.strip() or source != source.strip():
        raise ValueError("resident motion binding source is invalid")
    return {"source": source, "mode": ControlMode(value["mode"]).value}


def _validate_error(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _ERROR_FIELDS:
        raise ValueError("resident motion error fields are invalid")
    code = value["code"]
    message = value["message"]
    if not isinstance(code, str) or code not in _ERROR_CODES:
        raise ValueError("resident motion error code is invalid")
    if not isinstance(message, str) or not message:
        raise ValueError("resident motion error message is invalid")
    return {"code": code, "message": message}


def _uint64(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 0xFFFF_FFFF_FFFF_FFFF
    ):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _optional_uint64(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _uint64(name, value)


def _optional_segment_max_steps(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 2000
    ):
        raise ValueError(f"{name} must be null or within [1, 2000]")
    return value


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


__all__ = [
    "MAX_CONTROL_FRAME_BYTES",
    "SCHEMA_VERSION",
    "ResidentMotionControlServer",
    "main",
    "request_resident_motion_control",
]


if __name__ == "__main__":
    raise SystemExit(main())
