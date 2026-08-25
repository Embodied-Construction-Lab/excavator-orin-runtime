from __future__ import annotations

from dataclasses import dataclass, replace
import io

import pytest

import edge_runtime.resident_fixed_cycle_control as control
from edge_runtime.resident_fixed_cycle_control import (
    ResidentFixedCycleControlServer,
    request_resident_fixed_cycle_control,
)


@dataclass(frozen=True)
class _Snapshot:
    run_id: str = ""
    stage: str = "IDLE"
    requested_cycles: int = 0
    completed_cycles: int = 0
    current_dig_point_id: str = ""
    terminal: bool = False
    outcome: str = ""
    reason_code: str = ""


class _Runtime:
    def __init__(self) -> None:
        self.snapshot = _Snapshot()
        self.calls = []

    def start(self, *, run_id, requested_cycles, first_dig_point_id=None):
        self.calls.append(("start", run_id, requested_cycles, first_dig_point_id))
        self.snapshot = _Snapshot(
            run_id=run_id,
            stage="FOLLOW_DIG",
            requested_cycles=requested_cycles,
            current_dig_point_id=first_dig_point_id or "dig_01",
        )
        return self.snapshot

    def cancel(self):
        self.calls.append(("cancel",))
        self.snapshot = replace(
            self.snapshot,
            stage="CANCELLED",
            terminal=True,
            outcome="CANCELLED",
            reason_code="CANCELLED",
        )

    def heartbeat(self):
        self.calls.append(("heartbeat",))

    @property
    def visualization_snapshot(self):
        if self.snapshot.stage != "FOLLOW_DIG":
            return None
        return {
            "frame_id": "machine_root_ros",
            "target_id": self.snapshot.current_dig_point_id,
            "waypoints": (
                (0.8, 0.2, -0.1),
                (0.9, 0.1, -0.05),
                (1.0, 0.0, 0.0),
            ),
            "current_waypoint_index": 1,
            "waypoint_tolerance_m": 0.40,
        }


def test_start_status_and_cancel_use_one_strict_local_control_socket(tmp_path) -> None:
    runtime = _Runtime()
    path = tmp_path / "fixed-cycle.sock"
    server = ResidentFixedCycleControlServer(runtime, socket_path=str(path))
    server.start()
    try:
        started = request_resident_fixed_cycle_control(
            str(path),
            "start",
            run_id="field-run-001",
            requested_cycles=3,
            first_dig_point_id="dig_02",
        )
        status = request_resident_fixed_cycle_control(str(path), "status")
        cancelled = request_resident_fixed_cycle_control(str(path), "cancel")
    finally:
        server.close()

    assert started["ok"] is True
    assert started["status"]["stage"] == "FOLLOW_DIG"
    assert started["status"]["active_trajectory"] == {
        "frame_id": "machine_root_ros",
        "target_id": "dig_02",
        "waypoints": [
            [0.8, 0.2, -0.1],
            [0.9, 0.1, -0.05],
            [1.0, 0.0, 0.0],
        ],
        "current_waypoint_index": 1,
        "waypoint_tolerance_m": 0.40,
    }
    assert status["status"]["current_dig_point_id"] == "dig_02"
    assert cancelled["status"]["stage"] == "CANCELLED"
    assert runtime.calls == [
        ("start", "field-run-001", 3, "dig_02"),
        ("cancel",),
    ]
    assert not path.exists()


def test_client_rejects_command_specific_field_misuse_before_connect(tmp_path) -> None:
    path = str(tmp_path / "missing.sock")

    with pytest.raises(ValueError, match="only valid for start"):
        request_resident_fixed_cycle_control(
            path,
            "status",
            run_id="bad",
            requested_cycles=1,
        )
    with pytest.raises(ValueError, match="requested_cycles"):
        request_resident_fixed_cycle_control(
            path,
            "start",
            run_id="field-run-001",
            requested_cycles=10,
        )


def test_heartbeat_renews_liveness_without_starting_a_stage(tmp_path) -> None:
    runtime = _Runtime()
    path = tmp_path / "fixed-cycle.sock"
    server = ResidentFixedCycleControlServer(runtime, socket_path=str(path))
    server.start()
    try:
        response = request_resident_fixed_cycle_control(str(path), "heartbeat")
    finally:
        server.close()

    assert response["ok"] is True
    assert runtime.calls == [("heartbeat",)]


def test_runtime_failure_is_sanitized_and_does_not_stop_the_listener(tmp_path) -> None:
    class BrokenRuntime(_Runtime):
        def start(self, **_kwargs):
            raise RuntimeError("secret internal filesystem detail")

    path = tmp_path / "fixed-cycle.sock"
    server = ResidentFixedCycleControlServer(
        BrokenRuntime(), socket_path=str(path)
    )
    server.start()
    try:
        failed = request_resident_fixed_cycle_control(
            str(path),
            "start",
            run_id="field-run-001",
            requested_cycles=1,
        )
        status = request_resident_fixed_cycle_control(str(path), "status")
    finally:
        server.close()

    assert failed["ok"] is False
    assert failed["error"] == {
        "code": "command_failed",
        "message": "start was rejected by the resident fixed cycle",
    }
    assert "secret" not in str(failed)
    assert status["ok"] is True


def test_control_server_lifecycle_and_socket_path_fail_closed(tmp_path) -> None:
    path = tmp_path / "fixed-cycle.sock"
    path.write_text("not a socket", encoding="utf-8")
    server = ResidentFixedCycleControlServer(_Runtime(), socket_path=str(path))
    with pytest.raises(FileExistsError):
        server.start()

    path.unlink()
    server.start()
    with pytest.raises(RuntimeError, match="already running"):
        server.start()
    server.close()
    server.close()


@pytest.mark.parametrize(
    "timeout",
    [False, 0.0, 31.0, "1"],
)
def test_client_rejects_invalid_timeout_before_connect(tmp_path, timeout) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        request_resident_fixed_cycle_control(
            str(tmp_path / "missing.sock"),
            "status",
            timeout_s=timeout,
        )


@pytest.mark.parametrize(
    ("command", "kwargs", "message"),
    [
        ("unknown", {}, "unsupported"),
        ("start", {"run_id": "bad id", "requested_cycles": 1}, "run_id"),
        ("start", {"run_id": "run", "requested_cycles": True}, "cycles"),
    ],
)
def test_request_builder_rejects_invalid_commands(command, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        control._request(
            command,
            run_id=kwargs.get("run_id"),
            requested_cycles=kwargs.get("requested_cycles"),
            first_dig_point_id=kwargs.get("first_dig_point_id"),
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"schema_version": control.SCHEMA_VERSION, "command": "unknown"},
        {
            "schema_version": control.SCHEMA_VERSION,
            "command": "start",
            "run_id": "run",
            "requested_cycles": 1,
        },
    ],
)
def test_server_request_validation_is_strict(value) -> None:
    with pytest.raises(ValueError, match="request"):
        control._validate_request(value)


def test_cli_outputs_stable_json_and_sanitizes_connection_failure(
    monkeypatch, tmp_path
) -> None:
    response = {
        "schema_version": control.SCHEMA_VERSION,
        "ok": True,
        "command": "status",
        "status": control._status(_Snapshot()),
        "error": None,
    }
    monkeypatch.setattr(
        control,
        "request_resident_fixed_cycle_control",
        lambda *_args, **_kwargs: response,
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    assert control.main(
        ["--socket", str(tmp_path / "control.sock"), "status"],
        stdout=stdout,
        stderr=stderr,
    ) == 0
    assert json_loads(stdout.getvalue())["status"]["stage"] == "IDLE"
    assert stderr.getvalue() == ""

    monkeypatch.setattr(
        control,
        "request_resident_fixed_cycle_control",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("secret path")),
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    assert control.main(
        ["--socket", str(tmp_path / "control.sock"), "status"],
        stdout=stdout,
        stderr=stderr,
    ) == 2
    assert "secret" not in stderr.getvalue()


def test_cli_reports_rejected_response_without_internal_details(
    monkeypatch, tmp_path
) -> None:
    response = {
        "schema_version": control.SCHEMA_VERSION,
        "ok": False,
        "command": "cancel",
        "status": None,
        "error": {"code": "command_failed", "message": "hidden"},
    }
    monkeypatch.setattr(
        control,
        "request_resident_fixed_cycle_control",
        lambda *_args, **_kwargs: response,
    )
    stderr = io.StringIO()
    assert control.main(
        ["--socket", str(tmp_path / "control.sock"), "cancel"],
        stdout=io.StringIO(),
        stderr=stderr,
    ) == 2
    assert stderr.getvalue().strip() == "cancel failed: command_failed"


@pytest.mark.parametrize(
    "value",
    [
        {},
        {
            "schema_version": "wrong",
            "ok": True,
            "command": "status",
            "status": {},
            "error": None,
        },
        {
            "schema_version": control.SCHEMA_VERSION,
            "ok": True,
            "command": "unknown",
            "status": {},
            "error": None,
        },
        {
            "schema_version": control.SCHEMA_VERSION,
            "ok": False,
            "command": "status",
            "status": {},
            "error": {},
        },
    ],
)
def test_response_validation_rejects_invalid_envelopes(value) -> None:
    with pytest.raises(ValueError, match="response"):
        control._validate_response(value)


def json_loads(value: str):
    import json

    return json.loads(value)
