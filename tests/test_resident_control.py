import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

from edge_runtime.resident_control import (
    DEFAULT_MISSION_LEASE_MS,
    MAX_CONTROL_FRAME_BYTES,
    ResidentMotionControlServer,
    main,
    request_resident_motion_control,
)
from edge_runtime.resident_core import ResidentMotionCore
from edge_runtime.resident_motion import ControlMode, HandoffPhase, PolicyBinding


def test_default_mission_lease_tolerates_one_transient_control_rpc_delay() -> None:
    assert DEFAULT_MISSION_LEASE_MS == 3000


SCHEMA_VERSION = "resident_motion_control.v1"


class FakeCore:
    def __init__(self) -> None:
        self.phase = HandoffPhase.IDLE
        self.generation = 0
        self.active_binding = None
        self.target_binding = None
        self.last_handoff_latency_ms = None
        self.is_operational = True
        self.rl_is_active = False
        self.act_is_active = False
        self.mission_lease_is_active = False
        self._act_segment_generation = None
        self._act_segment_max_steps = None
        self._act_segment_completed_steps = 0
        self._act_segment_complete = False
        self.calls = []
        self.terminal_disarm_calls = 0
        self.inflight = 0
        self.max_inflight = 0
        self.lock = threading.Lock()

    def snapshot(self):
        return SimpleNamespace(
            generation=self.generation,
            phase=self.phase,
            active_binding=self.active_binding,
            target_binding=self.target_binding,
            last_handoff_latency_ms=self.last_handoff_latency_ms,
        )

    def act_segment_snapshot(self):
        return SimpleNamespace(
            generation=self._act_segment_generation,
            max_steps=self._act_segment_max_steps,
            completed_steps=self._act_segment_completed_steps,
            complete=self._act_segment_complete,
        )

    def _enter(self, command):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        time.sleep(0.01)
        self.calls.append(command)
        self.generation += 1
        with self.lock:
            self.inflight -= 1

    def activate_rl(self):
        self._enter("activate_rl")
        self.phase = HandoffPhase.TARGET_ZERO_PENDING
        self.target_binding = PolicyBinding(
            "rl_follow", ControlMode.VELOCITY_REFERENCE
        )
        return self.generation

    def renew_mission_lease(self, *, lease_ms):
        self.calls.append(("renew_lease", lease_ms))
        self.mission_lease_is_active = True

    def activate_act(self, *, max_steps):
        self._enter("activate_act")
        self._act_segment_generation = (self._act_segment_generation or 0) + 1
        self._act_segment_max_steps = max_steps
        self._act_segment_completed_steps = 0
        self._act_segment_complete = False
        self.phase = HandoffPhase.TARGET_ZERO_PENDING
        self.target_binding = PolicyBinding("act_dig", ControlMode.MANUAL_ACTION)
        return self.generation

    def terminal_disarm(self):
        self._enter("terminal_disarm")
        self.terminal_disarm_calls += 1
        self.phase = HandoffPhase.IDLE
        self.active_binding = None
        self.target_binding = None
        self.is_operational = False


def send_raw_request(path: str, payload: bytes) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1.0)
        connection.connect(path)
        connection.sendall(len(payload).to_bytes(4, "big") + payload)
        header = connection.recv(4)
        size = int.from_bytes(header, "big")
        body = bytearray()
        while len(body) < size:
            body.extend(connection.recv(size - len(body)))
    return json.loads(bytes(body).decode("utf-8"))


class ResidentMotionControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = str(Path(self.temp_dir.name) / "motion-control.sock")
        self.core = FakeCore()
        self.server = ResidentMotionControlServer(self.core, socket_path=self.path)

    def start(self) -> None:
        self.server.start()
        self.addCleanup(self.server.close)

    def test_status_round_trip_has_exact_typed_contract(self) -> None:
        self.core.phase = HandoffPhase.ACTIVE
        self.core.generation = 9
        self.core.active_binding = PolicyBinding(
            "rl_follow", ControlMode.VELOCITY_REFERENCE
        )
        self.core.last_handoff_latency_ms = 37.25
        self.core.rl_is_active = True
        self.start()

        response = request_resident_motion_control(self.path, "status")

        self.assertEqual(
            set(response),
            {"schema_version", "ok", "command", "status", "error"},
        )
        self.assertEqual(response["schema_version"], SCHEMA_VERSION)
        self.assertTrue(response["ok"])
        self.assertEqual(response["command"], "status")
        self.assertIsNone(response["error"])
        self.assertEqual(
            response["status"],
            {
                "phase": "active",
                "control_generation": 9,
                "active": {
                    "source": "rl_follow",
                    "mode": "velocity_reference",
                },
                "target": None,
                "last_handoff_latency_ms": 37.25,
                "rl_is_active": True,
                "act_is_active": False,
                "act_worker_ready": True,
                "act_segment_generation": None,
                "act_segment_max_steps": None,
                "act_segment_completed_steps": 0,
                "act_segment_complete": False,
                "mission_lease_active": False,
                "is_operational": True,
            },
        )
        self.assertEqual(self.core.calls, [])

    def test_renew_lease_is_strict_and_does_not_change_generation(self) -> None:
        self.start()

        response = request_resident_motion_control(self.path, "renew_lease")

        self.assertTrue(response["ok"])
        self.assertEqual(response["status"]["control_generation"], 0)
        self.assertTrue(response["status"]["mission_lease_active"])
        self.assertEqual(
            self.core.calls,
            [("renew_lease", DEFAULT_MISSION_LEASE_MS)],
        )

    def test_motion_activation_arms_the_same_bounded_mission_lease(self) -> None:
        self.start()

        response = request_resident_motion_control(self.path, "activate_act", max_steps=130)

        self.assertTrue(response["ok"])
        self.assertTrue(response["status"]["mission_lease_active"])
        self.assertEqual(
            self.core.calls[:2],
            [("renew_lease", DEFAULT_MISSION_LEASE_MS), "activate_act"],
        )

    def test_status_reads_real_core_segment_snapshot_atomically(self) -> None:
        class RecordingSerial:
            def write(self, payload):
                return len(payload)

            def flush(self):
                return None

        core = ResidentMotionCore(RecordingSerial(), max_state_age_ms=200.0)
        server = ResidentMotionControlServer(core, socket_path=self.path)
        server.start()
        self.addCleanup(server.close)

        response = request_resident_motion_control(self.path, "status")

        self.assertTrue(response["ok"])
        self.assertIsNone(response["status"]["act_segment_generation"])
        self.assertIsNone(response["status"]["act_segment_max_steps"])
        self.assertEqual(response["status"]["act_segment_completed_steps"], 0)
        self.assertFalse(response["status"]["act_segment_complete"])

    def test_disconnected_act_worker_rejects_activation_before_core_mutation(self) -> None:
        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            act_worker_ready=lambda: False,
        )
        server.start()
        self.addCleanup(server.close)

        status = request_resident_motion_control(self.path, "status")
        rejected = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertTrue(status["ok"])
        self.assertFalse(status["status"]["act_worker_ready"])
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["command"], "activate_act")
        self.assertEqual(rejected["error"]["code"], "command_failed")
        self.assertEqual(self.core.calls, [])
        self.assertEqual(self.core.generation, 0)

    def test_connected_act_worker_allows_activation_and_reports_ready(self) -> None:
        ready = False
        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            act_worker_ready=lambda: ready,
        )
        server.start()
        self.addCleanup(server.close)

        rejected = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )
        ready = True
        activated = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertFalse(rejected["ok"])
        self.assertTrue(activated["ok"])
        self.assertTrue(activated["status"]["act_worker_ready"])
        self.assertEqual(
            self.core.calls,
            [("renew_lease", DEFAULT_MISSION_LEASE_MS), "activate_act"],
        )
        self.assertEqual(activated["status"]["control_generation"], 1)
        self.assertEqual(activated["status"]["act_segment_generation"], 1)
        self.assertEqual(activated["status"]["act_segment_max_steps"], 130)
        self.assertEqual(activated["status"]["act_segment_completed_steps"], 0)
        self.assertFalse(activated["status"]["act_segment_complete"])

    def test_busy_rl_behavior_rejects_act_before_core_generation_changes(self) -> None:
        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            rl_behavior_idle=lambda: False,
        )
        server.start()
        self.addCleanup(server.close)

        rejected = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["command"], "activate_act")
        self.assertEqual(rejected["error"]["code"], "command_failed")
        self.assertEqual(self.core.calls, [])
        self.assertEqual(self.core.generation, 0)

    def test_idle_rl_behavior_allows_act_activation(self) -> None:
        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            rl_behavior_idle=lambda: True,
        )
        server.start()
        self.addCleanup(server.close)

        activated = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertTrue(activated["ok"])
        self.assertEqual(
            self.core.calls,
            [("renew_lease", DEFAULT_MISSION_LEASE_MS), "activate_act"],
        )
        self.assertEqual(self.core.generation, 1)

    def test_atomic_rl_idle_gate_wraps_lease_and_act_activation(self) -> None:
        events = []

        def run_when_idle(operation):
            events.append("gate_enter")
            result = operation()
            events.append("gate_exit")
            return result

        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            activate_act_while_rl_idle=run_when_idle,
            rl_behavior_idle=lambda: (_ for _ in ()).throw(
                AssertionError("non-atomic probe must not be used")
            ),
        )
        server.start()
        self.addCleanup(server.close)

        activated = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertTrue(activated["ok"])
        self.assertEqual(events, ["gate_enter", "gate_exit"])
        self.assertEqual(
            self.core.calls,
            [("renew_lease", DEFAULT_MISSION_LEASE_MS), "activate_act"],
        )

    def test_atomic_rl_idle_gate_rejection_never_renews_or_mutates_core(self) -> None:
        def reject(_operation):
            raise RuntimeError("RL behavior is active")

        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            activate_act_while_rl_idle=reject,
        )
        server.start()
        self.addCleanup(server.close)

        rejected = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertFalse(rejected["ok"])
        self.assertEqual(self.core.calls, [])

    def test_broken_or_nonboolean_rl_behavior_probe_fails_closed(self) -> None:
        def broken_probe():
            raise RuntimeError("remote behavior state unavailable")

        for index, probe in enumerate((broken_probe, lambda: 1, lambda: None)):
            with self.subTest(probe=index):
                path = str(
                    Path(self.temp_dir.name) / f"rl-behavior-probe-{index}.sock"
                )
                server = ResidentMotionControlServer(
                    self.core,
                    socket_path=path,
                    rl_behavior_idle=probe,
                )
                server.start()
                self.addCleanup(server.close)

                rejected = request_resident_motion_control(
                    path, "activate_act", max_steps=130
                )

                self.assertFalse(rejected["ok"])
                self.assertEqual(rejected["error"]["code"], "command_failed")
                self.assertEqual(self.core.calls, [])
                self.assertEqual(self.core.generation, 0)

    def test_rl_behavior_idle_probe_must_be_callable(self) -> None:
        with self.assertRaisesRegex(ValueError, "rl_behavior_idle"):
            ResidentMotionControlServer(
                self.core,
                socket_path=self.path,
                rl_behavior_idle=True,
            )

    def test_broken_or_nonboolean_act_probe_fails_closed(self) -> None:
        def broken_probe():
            raise OSError("worker socket disappeared")

        for index, probe in enumerate((broken_probe, lambda: 1)):
            path = str(Path(self.temp_dir.name) / f"probe-{index}.sock")
            server = ResidentMotionControlServer(
                self.core,
                socket_path=path,
                act_worker_ready=probe,
            )
            server.start()
            self.addCleanup(server.close)

            status = request_resident_motion_control(path, "status")
            rejected = request_resident_motion_control(
                path, "activate_act", max_steps=130
            )

            self.assertFalse(status["status"]["act_worker_ready"])
            self.assertFalse(rejected["ok"])
        self.assertEqual(self.core.calls, [])

    def test_only_explicit_commands_mutate_core_and_close_does_not_disarm(self) -> None:
        self.start()
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

        rl = request_resident_motion_control(self.path, "activate_rl")
        act = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertTrue(rl["ok"])
        self.assertEqual(rl["status"]["control_generation"], 1)
        self.assertEqual(rl["status"]["target"]["source"], "rl_follow")
        self.assertTrue(act["ok"])
        self.assertEqual(act["status"]["control_generation"], 2)
        self.assertEqual(act["status"]["target"]["source"], "act_dig")
        self.server.close()
        self.assertEqual(self.core.terminal_disarm_calls, 0)
        self.assertFalse(os.path.exists(self.path))

        self.server = ResidentMotionControlServer(self.core, socket_path=self.path)
        self.start()
        disarmed = request_resident_motion_control(self.path, "terminal_disarm")
        self.assertTrue(disarmed["ok"])
        self.assertFalse(disarmed["status"]["is_operational"])
        self.assertEqual(self.core.terminal_disarm_calls, 1)

    def test_same_server_instance_can_restart_without_reviving_old_accept_loop(self) -> None:
        self.server.start()
        first = request_resident_motion_control(self.path, "status")
        self.server.close()

        self.server.start()
        self.addCleanup(self.server.close)
        second = request_resident_motion_control(self.path, "status")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual(self.core.calls, [])

    def test_request_schema_is_exact_and_invalid_input_never_reaches_core(self) -> None:
        self.start()
        invalid_payloads = (
            b'{}',
            b'{"schema_version":"wrong","command":"status"}',
            b'{"schema_version":"resident_motion_control.v1","command":"move"}',
            b'{"schema_version":"resident_motion_control.v1","command":"activate_act"}',
            b'{"schema_version":"resident_motion_control.v1","command":"activate_act","max_steps":0}',
            b'{"schema_version":"resident_motion_control.v1","command":"activate_act","max_steps":2001}',
            b'{"schema_version":"resident_motion_control.v1","command":"activate_act","max_steps":true}',
            b'{"schema_version":"resident_motion_control.v1","command":"status","max_steps":130}',
            b'{"schema_version":"resident_motion_control.v1","command":"status","extra":1}',
            b'{"schema_version":"resident_motion_control.v1","command":"status","command":"activate_act"}',
            b'not-json',
            b'\xff',
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = send_raw_request(self.path, payload)
                self.assertFalse(response["ok"])
                self.assertIsNone(response["status"])
                self.assertEqual(response["error"]["code"], "invalid_request")
                self.assertTrue(response["error"]["message"])

        self.assertEqual(self.core.calls, [])

    def test_four_byte_big_endian_frame_and_4096_byte_limit(self) -> None:
        self.start()
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "command": "status"},
            separators=(",", ":"),
        ).encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1.0)
            connection.connect(self.path)
            connection.sendall(len(payload).to_bytes(4, "big") + payload)
            header = connection.recv(4)
            self.assertEqual(len(header), 4)
            response_size = int.from_bytes(header, "big")
            self.assertLessEqual(response_size, MAX_CONTROL_FRAME_BYTES)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1.0)
            connection.connect(self.path)
            connection.sendall((MAX_CONTROL_FRAME_BYTES + 1).to_bytes(4, "big"))
            header = connection.recv(4)
            response_size = int.from_bytes(header, "big")
            body = connection.recv(response_size)
        response = json.loads(body.decode("utf-8"))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertEqual(self.core.calls, [])

    def test_absolute_path_is_required_and_existing_path_is_not_removed(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            ResidentMotionControlServer(self.core, socket_path="relative.sock")
        with self.assertRaisesRegex(ValueError, "absolute"):
            request_resident_motion_control("relative.sock", "status")

        Path(self.path).write_text("do-not-delete", encoding="utf-8")
        server = ResidentMotionControlServer(self.core, socket_path=self.path)
        with self.assertRaises(FileExistsError):
            server.start()
        server.close()
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "do-not-delete")

    def test_stale_socket_is_reclaimed_but_active_listener_is_preserved(self) -> None:
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(self.path)
        stale.close()

        reclaimed = ResidentMotionControlServer(self.core, socket_path=self.path)
        reclaimed.start()
        self.addCleanup(reclaimed.close)
        self.assertTrue(request_resident_motion_control(self.path, "status")["ok"])
        reclaimed.close()

        active_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        active_listener.bind(self.path)
        active_listener.listen()
        self.addCleanup(active_listener.close)
        conflicting = ResidentMotionControlServer(self.core, socket_path=self.path)
        with self.assertRaisesRegex(RuntimeError, "already in use"):
            conflicting.start()
        conflicting.close()
        self.assertTrue(Path(self.path).exists())

    def test_concurrent_clients_are_dispatched_serially(self) -> None:
        self.start()
        responses = []
        failures = []

        def request(command):
            try:
                responses.append(
                    request_resident_motion_control(
                        self.path,
                        command,
                        max_steps=130 if command == "activate_act" else None,
                        timeout_s=2.0,
                    )
                )
            except Exception as exc:  # pragma: no cover - assertion captures details
                failures.append(exc)

        clients = [
            threading.Thread(
                target=request,
                args=("activate_rl" if index % 2 == 0 else "activate_act",),
            )
            for index in range(8)
        ]
        for client in clients:
            client.start()
        for client in clients:
            client.join()

        self.assertEqual(failures, [])
        self.assertEqual(len(responses), 8)
        self.assertEqual(self.core.max_inflight, 1)
        self.assertEqual(
            sorted(response["status"]["control_generation"] for response in responses),
            list(range(1, 9)),
        )

    def test_core_failure_is_readable_but_does_not_leak_exception_details(self) -> None:
        class FailingCore(FakeCore):
            def activate_act(self, *, max_steps):
                raise RuntimeError("secret=/home/operator/private-token")

        server = ResidentMotionControlServer(FailingCore(), socket_path=self.path)
        server.start()
        self.addCleanup(server.close)

        response = request_resident_motion_control(
            self.path, "activate_act", max_steps=130
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "command_failed")
        self.assertIn("activate_act", response["error"]["message"])
        self.assertNotIn("secret", response["error"]["message"])
        self.assertNotIn("private-token", json.dumps(response))

    def test_core_value_error_is_a_command_failure_not_an_invalid_request(self) -> None:
        class RejectingCore(FakeCore):
            def activate_rl(self):
                raise ValueError("handoff cannot start")

        server = ResidentMotionControlServer(RejectingCore(), socket_path=self.path)
        server.start()
        self.addCleanup(server.close)

        response = request_resident_motion_control(self.path, "activate_rl")

        self.assertFalse(response["ok"])
        self.assertEqual(response["command"], "activate_rl")
        self.assertEqual(response["error"]["code"], "command_failed")

    def test_client_rejects_invalid_commands_before_connecting(self) -> None:
        for invalid in ("", "move", None, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    request_resident_motion_control(self.path, invalid)

        invalid_tagged_requests = (
            ("activate_act", None),
            ("activate_act", 0),
            ("activate_act", 2001),
            ("activate_act", True),
            ("status", 130),
            ("activate_rl", 130),
            ("terminal_disarm", 130),
        )
        for command, max_steps in invalid_tagged_requests:
            with self.subTest(command=command, max_steps=max_steps):
                with self.assertRaises(ValueError):
                    request_resident_motion_control(
                        self.path,
                        command,
                        max_steps=max_steps,
                    )

    def test_module_cli_prints_stable_json_and_exits_zero(self) -> None:
        self.start()

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_runtime.resident_control",
                "--socket",
                self.path,
                "status",
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=2.0,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        decoded = json.loads(completed.stdout)
        self.assertTrue(decoded["ok"])
        self.assertEqual(decoded["command"], "status")
        self.assertEqual(
            completed.stdout,
            json.dumps(
                decoded,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )

    def test_module_cli_reports_command_failure_without_server_details(self) -> None:
        server = ResidentMotionControlServer(
            self.core,
            socket_path=self.path,
            act_worker_ready=lambda: False,
        )
        server.start()
        self.addCleanup(server.close)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_runtime.resident_control",
                "--socket",
                self.path,
                "activate_act",
                "--max-steps",
                "130",
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=2.0,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "activate_act failed: command_failed\n")
        self.assertNotIn(self.path, completed.stderr)
        self.assertEqual(self.core.calls, [])

    def test_module_cli_activate_act_passes_required_step_budget(self) -> None:
        self.start()

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_runtime.resident_control",
                "--socket",
                self.path,
                "activate_act",
                "--max-steps",
                "130",
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=2.0,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = json.loads(completed.stdout)
        self.assertEqual(response["command"], "activate_act")
        self.assertEqual(response["status"]["act_segment_max_steps"], 130)
        self.assertEqual(
            self.core.calls,
            [("renew_lease", DEFAULT_MISSION_LEASE_MS), "activate_act"],
        )

    def test_cli_connection_and_invalid_response_errors_are_sanitized(self) -> None:
        cases = (
            OSError(f"cannot connect to {self.path}; secret-token"),
            ValueError(f"invalid response from {self.path}; secret-token"),
        )
        for error in cases:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.subTest(error=type(error).__name__), mock.patch(
                "edge_runtime.resident_control.request_resident_motion_control",
                side_effect=error,
            ):
                exit_code = main(
                    ["--socket", self.path, "status"],
                    stdout=stdout,
                    stderr=stderr,
                )

            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(), "resident motion control request failed\n"
            )
            self.assertNotIn(self.path, stderr.getvalue())
            self.assertNotIn("secret-token", stderr.getvalue())

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch(
            "edge_runtime.resident_control.request_resident_motion_control",
            return_value={"ok": "not-a-boolean"},
        ):
            exit_code = main(
                ["--socket", self.path, "status"],
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr.getvalue(), "resident motion control request failed\n")

        stdout = io.StringIO()
        stderr = io.StringIO()
        malicious_error = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "command": "status",
            "status": None,
            "error": {
                "code": "secret-token\nterminal-injection",
                "message": "hidden",
            },
        }
        with mock.patch(
            "edge_runtime.resident_control.request_resident_motion_control",
            return_value=malicious_error,
        ):
            exit_code = main(
                ["--socket", self.path, "status"],
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(stderr.getvalue(), "resident motion control request failed\n")
        self.assertNotIn("secret-token", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
