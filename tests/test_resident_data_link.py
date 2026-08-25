import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
import unittest

from edge_runtime.resident_data_link import (
    MAX_FRAME_BYTES,
    ResidentActDataLink,
)
from edge_runtime.resident_core import ResidentMotionCore
from edge_runtime.resident_motion import ControlMode, MotionCandidate, ZERO_ACTION
from edge_runtime.resident_protocol import encode_motion_candidate
from edge_runtime.resident_sink import ResidentTelemetry, ResidentWriteResult
from edge_runtime.resident_state import (
    ResidentActState,
    decode_resident_state,
    encode_resident_state,
)


def _state(*, control_seq: int = 7) -> ResidentActState:
    return ResidentActState(
        state=(1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        receive_monotonic_ns=2_000,
        state_monotonic_ns=1_900,
        control_seq=control_seq,
        sensor_seq=11,
        sensor_is_new=True,
        control_enabled=True,
        estop=False,
        rs485_ok=True,
        dwj_ok=True,
        imu_ok=True,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
        control_generation=4,
    )


def _candidate(
    *,
    first: float = 0.1,
    generation: int = 4,
    source: str = "act",
    created_monotonic_ns: int = 2_100,
    valid_until_monotonic_ns: int = 3_100,
) -> bytes:
    return encode_motion_candidate(
        MotionCandidate(
            source=source,
            generation=generation,
            mode=ControlMode.MANUAL_ACTION,
            action=(first, 0.2, 0.3, 0.0),
            created_monotonic_ns=created_monotonic_ns,
            valid_until_monotonic_ns=valid_until_monotonic_ns,
        )
    )


class RecordingSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None


def _telemetry(
    *,
    receive_ns: int,
    command_seq: int,
    valid: bool,
    mode: ControlMode | None,
) -> ResidentTelemetry:
    return ResidentTelemetry(
        receive_monotonic_ns=receive_ns,
        command_rx_seq=command_seq,
        command_valid=valid,
        command_timed_out=False,
        control_mode=mode,
        command_action=ZERO_ACTION,
        control_enabled=True,
        estop=False,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
    )


def _send_frame(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(len(payload).to_bytes(4, "big") + payload)


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("connection closed while receiving a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: socket.socket) -> bytes:
    header = _recv_exact(connection, 4)
    length = int.from_bytes(header, "big")
    if not 0 < length <= MAX_FRAME_BYTES:
        raise ValueError("invalid test frame length")
    return _recv_exact(connection, length)


def _wait_until(predicate, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class ResidentActDataLinkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.socket_path = (
            Path(self.temporary_directory.name) / "runtime" / "act-worker.sock"
        )
        self.received: list[bytes] = []
        self.received_event = threading.Event()

        def receive_candidate(payload: bytes) -> None:
            self.received.append(payload)
            self.received_event.set()

        self.link = ResidentActDataLink(
            self.socket_path,
            on_candidate=receive_candidate,
        )
        self.addCleanup(self.link.close)

    def connect(self) -> socket.socket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(1.0)
        connection.connect(os.fspath(self.socket_path))
        self.addCleanup(connection.close)
        self.assertTrue(_wait_until(lambda: self.link.connected))
        return connection

    def test_requires_an_absolute_socket_path_and_callback(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            ResidentActDataLink("relative.sock", on_candidate=lambda _payload: None)
        with self.assertRaisesRegex(ValueError, "callback"):
            ResidentActDataLink(self.socket_path, on_candidate=None)  # type: ignore[arg-type]

    def test_start_creates_private_parent_and_close_removes_only_owned_socket(self) -> None:
        self.assertFalse(self.socket_path.parent.exists())
        self.assertFalse(self.link.ready)

        self.link.start()

        self.assertTrue(self.link.ready)
        self.assertTrue(self.socket_path.parent.is_dir())
        self.assertTrue(stat.S_ISSOCK(self.socket_path.lstat().st_mode))
        self.assertEqual(stat.S_IMODE(self.socket_path.lstat().st_mode), 0o600)
        connection = self.connect()
        self.assertTrue(self.link.connected)
        connection.close()
        self.assertTrue(_wait_until(lambda: not self.link.connected))

        self.link.close()

        self.assertFalse(self.link.ready)
        self.assertFalse(self.link.connected)
        self.assertFalse(self.socket_path.exists())

    def test_bidirectional_framing_validates_both_protocols(self) -> None:
        self.link.start()
        encoded_state = encode_resident_state(_state())
        self.link.publish(encoded_state)
        worker = self.connect()

        outbound = _recv_frame(worker)
        self.assertEqual(decode_resident_state(outbound), _state())

        inbound = _candidate()
        framed = len(inbound).to_bytes(4, "big") + inbound
        for octet in framed:
            worker.sendall(bytes((octet,)))
        self.assertTrue(self.received_event.wait(1.0))
        self.assertEqual(self.received, [inbound])

    def test_publish_is_latest_only_while_no_worker_is_connected(self) -> None:
        self.link.start()
        for sequence in range(1, 101):
            self.link.publish(encode_resident_state(_state(control_seq=sequence)))

        worker = self.connect()

        newest = decode_resident_state(_recv_frame(worker))
        self.assertEqual(newest.control_seq, 100)
        worker.settimeout(0.05)
        with self.assertRaises(socket.timeout):
            worker.recv(1)

    def test_publish_rejects_invalid_and_oversize_state_without_disconnect(self) -> None:
        self.link.start()
        worker = self.connect()
        with self.assertRaises(ValueError):
            self.link.publish(b"{}")

        valid = encode_resident_state(_state())
        exact_limit = valid + b" " * (MAX_FRAME_BYTES - len(valid))
        self.link.publish(exact_limit)
        self.assertEqual(len(_recv_frame(worker)), MAX_FRAME_BYTES)

        with self.assertRaises(ValueError):
            self.link.publish(exact_limit + b" ")
        self.assertTrue(self.link.connected)

    def test_oversize_or_invalid_candidate_disconnects_only_the_worker(self) -> None:
        self.link.start()
        worker = self.connect()
        worker.sendall((MAX_FRAME_BYTES + 1).to_bytes(4, "big"))
        self.assertTrue(_wait_until(lambda: not self.link.connected))
        self.assertEqual(self.received, [])
        self.assertTrue(self.link.ready)

        replacement = self.connect()
        _send_frame(replacement, b"{}")
        self.assertTrue(_wait_until(lambda: not self.link.connected))
        self.assertEqual(self.received, [])
        self.assertTrue(self.link.ready)

    def test_active_worker_is_not_replaced_by_a_second_connection(self) -> None:
        self.link.start()
        first = self.connect()
        second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        second.settimeout(1.0)
        self.addCleanup(second.close)
        second.connect(os.fspath(self.socket_path))
        try:
            _send_frame(second, _candidate(first=0.9))
        except (BrokenPipeError, ConnectionResetError):
            # The server deliberately rejects newcomers while one worker is
            # active.  Under load it may close the socket before this optional
            # probe reaches the kernel; both timings express the same contract.
            pass
        self.assertTrue(_wait_until(lambda: self.link.rejected_connection_count == 1))
        time.sleep(0.02)
        self.assertEqual(self.received, [])

        _send_frame(first, _candidate(first=0.1))
        self.assertTrue(self.received_event.wait(1.0))
        decoded = json.loads(self.received[0])
        self.assertEqual(decoded["action"][0], 0.1)
        self.assertTrue(self.link.connected)

    def test_disconnect_allows_clean_reconnect_without_closing_server(self) -> None:
        self.link.start()
        first = self.connect()
        first.shutdown(socket.SHUT_RDWR)
        first.close()
        self.assertTrue(_wait_until(lambda: not self.link.connected))
        self.assertTrue(self.link.ready)

        second = self.connect()
        _send_frame(second, _candidate())
        self.assertTrue(self.received_event.wait(1.0))
        self.assertEqual(self.received, [_candidate()])

    def test_worker_disconnect_notifies_the_owner_once(self) -> None:
        disconnects: list[str] = []
        disconnected = threading.Event()

        def on_connection_lost() -> None:
            disconnects.append("lost")
            disconnected.set()

        link = ResidentActDataLink(
            self.socket_path,
            on_candidate=lambda _payload: None,
            on_connection_lost=on_connection_lost,
        )
        link.start()
        self.addCleanup(link.close)
        worker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        worker.settimeout(1.0)
        worker.connect(os.fspath(self.socket_path))
        self.addCleanup(worker.close)
        self.assertTrue(_wait_until(lambda: link.connected))

        worker.shutdown(socket.SHUT_RDWR)
        worker.close()

        self.assertTrue(disconnected.wait(1.0))
        self.assertEqual(disconnects, ["lost"])
        self.assertFalse(link.connected)
        self.assertTrue(link.ready)

    def test_callback_failure_closes_connection_and_keeps_listener_ready(self) -> None:
        calls: list[bytes] = []

        def failing_callback(payload: bytes) -> None:
            calls.append(payload)
            raise LookupError("injected ingress rejection")

        link = ResidentActDataLink(self.socket_path, on_candidate=failing_callback)
        link.start()
        self.addCleanup(link.close)
        worker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        worker.settimeout(1.0)
        worker.connect(os.fspath(self.socket_path))
        self.addCleanup(worker.close)
        self.assertTrue(_wait_until(lambda: link.connected))

        _send_frame(worker, _candidate())

        self.assertTrue(_wait_until(lambda: not link.connected))
        self.assertEqual(calls, [_candidate()])
        self.assertTrue(link.ready)

    def test_generation_mismatch_is_passively_rejected_without_disconnect_or_revocation(self) -> None:
        serial = RecordingSerial()
        core = ResidentMotionCore(
            serial,
            max_state_age_ms=200.0,
            monotonic_ns=lambda: 1_080_000_000,
        )
        core.initialize(
            _telemetry(
                receive_ns=990_000_000,
                command_seq=0,
                valid=False,
                mode=None,
            )
        )

        def acknowledge_latest_zero(*, mode: ControlMode, receive_ns: int) -> None:
            packet = json.loads(serial.writes[-1].decode("ascii"))
            core.observe_telemetry(
                _telemetry(
                    receive_ns=receive_ns,
                    command_seq=packet["command_seq"],
                    valid=True,
                    mode=mode,
                )
            )

        old_generation = core.activate_act(now_monotonic_ns=1_000_000_000)
        acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_010_000_000,
        )
        core.activate_rl(now_monotonic_ns=1_020_000_000)
        acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_030_000_000,
        )
        acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_040_000_000,
        )
        current_generation = core.activate_act(now_monotonic_ns=1_050_000_000)
        acknowledge_latest_zero(
            mode=ControlMode.VELOCITY_REFERENCE,
            receive_ns=1_060_000_000,
        )
        acknowledge_latest_zero(
            mode=ControlMode.MANUAL_ACTION,
            receive_ns=1_070_000_000,
        )
        self.assertGreater(current_generation, old_generation)
        self.assertTrue(core.act_is_active)

        results: list[ResidentWriteResult] = []
        candidate_seen = threading.Event()

        def submit_candidate(payload: bytes) -> ResidentWriteResult:
            try:
                result = core.submit_act(payload)
                results.append(result)
                return result
            finally:
                candidate_seen.set()

        link = ResidentActDataLink(
            Path(self.temporary_directory.name) / "runtime" / "core-worker.sock",
            on_candidate=submit_candidate,
            on_connection_lost=core.notify_act_worker_disconnected,
        )
        link.start()
        self.addCleanup(link.close)
        worker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        worker.settimeout(1.0)
        worker.connect(os.fspath(link.socket_path))
        self.addCleanup(worker.close)
        self.assertTrue(_wait_until(lambda: link.connected))
        writes_before_old_frame = len(serial.writes)

        _send_frame(
            worker,
            _candidate(
                generation=old_generation,
                source="act_dig",
                created_monotonic_ns=1_075_000_000,
                valid_until_monotonic_ns=1_200_000_000,
            ),
        )

        self.assertTrue(candidate_seen.wait(1.0))
        self.assertEqual(len(results), 1)
        old_result = results[-1]
        self.assertFalse(old_result.accepted)
        self.assertEqual(old_result.reason, "stale_generation")
        self.assertTrue(link.connected)
        self.assertTrue(core.act_is_active)
        self.assertEqual(core.active_act_generation, current_generation)
        self.assertEqual(len(serial.writes), writes_before_old_frame)

        candidate_seen.clear()
        _send_frame(
            worker,
            _candidate(
                generation=current_generation + 1,
                source="act_dig",
                created_monotonic_ns=1_075_000_000,
                valid_until_monotonic_ns=1_200_000_000,
            ),
        )

        self.assertTrue(candidate_seen.wait(1.0))
        self.assertEqual(len(results), 2)
        future_result = results[-1]
        self.assertFalse(future_result.accepted)
        self.assertEqual(future_result.reason, "stale_generation")
        self.assertTrue(link.connected)
        self.assertTrue(core.act_is_active)
        self.assertEqual(core.active_act_generation, current_generation)
        self.assertEqual(len(serial.writes), writes_before_old_frame)

        candidate_seen.clear()
        _send_frame(
            worker,
            _candidate(
                first=0.25,
                generation=current_generation,
                source="act_dig",
                created_monotonic_ns=1_075_000_000,
                valid_until_monotonic_ns=1_200_000_000,
            ),
        )

        self.assertTrue(candidate_seen.wait(1.0))
        self.assertEqual(len(results), 3)
        self.assertTrue(results[-1].accepted)
        self.assertTrue(link.connected)
        self.assertEqual(len(serial.writes), writes_before_old_frame + 1)

    def test_close_does_not_unlink_a_replacement_file(self) -> None:
        self.link.start()
        os.unlink(self.socket_path)
        self.socket_path.write_text("not-owned-by-link", encoding="utf-8")

        self.link.close()

        self.assertEqual(
            self.socket_path.read_text(encoding="utf-8"),
            "not-owned-by-link",
        )


if __name__ == "__main__":
    unittest.main()
