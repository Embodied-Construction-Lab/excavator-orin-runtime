import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from edge_runtime.resident_control import (
    ResidentMotionControlServer,
    request_resident_motion_control,
)
from edge_runtime.resident_core import (
    AxisManualActionDeadzone,
    ManualActionDeadzoneContract,
    ResidentMotionCore,
)
from edge_runtime.resident_motion import ControlMode, MotionCandidate, ZERO_ACTION
from edge_runtime.resident_protocol import encode_motion_candidate
from edge_runtime.resident_sink import ResidentTelemetry


class RecordingSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None


def telemetry(*, receive_ns, command_seq, valid, mode, action=ZERO_ACTION):
    return ResidentTelemetry(
        receive_monotonic_ns=receive_ns,
        command_rx_seq=command_seq,
        command_valid=valid,
        command_timed_out=False,
        control_mode=mode,
        command_action=action,
        control_enabled=True,
        estop=False,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
    )


def active_act_at_final_step():
    serial = RecordingSerial()
    core = ResidentMotionCore(
        serial,
        max_state_age_ms=200.0,
        manual_action_deadzone_contract=ManualActionDeadzoneContract(
            (AxisManualActionDeadzone(0.15, 0.15),) * 4
        ),
        wall_time_ms=lambda: 10_000,
        monotonic_ns=lambda: 1_090_000_000,
    )
    core.initialize(
        telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
    )
    generation = core.activate_act(max_steps=1, now_monotonic_ns=1_000_000_000)
    activation_zero = json.loads(serial.writes[-1].decode("ascii"))
    core.observe_telemetry(
        telemetry(
            receive_ns=1_020_000_000,
            command_seq=activation_zero["command_seq"],
            valid=True,
            mode=ControlMode.MANUAL_ACTION,
        )
    )
    action = (0.1, -0.2, 0.3, 0.0)
    final_write = core.submit_act(
        encode_motion_candidate(
            MotionCandidate(
                source="act_dig",
                generation=generation,
                mode=ControlMode.MANUAL_ACTION,
                action=action,
                created_monotonic_ns=1_025_000_000,
                valid_until_monotonic_ns=1_200_000_000,
            )
        )
    )
    if not final_write.accepted:
        raise AssertionError("ACT final step was not accepted")
    return core, action, final_write


class ResidentControlAtomicityTest(unittest.TestCase):
    def test_status_cannot_mix_pre_and_post_act_completion_state(self) -> None:
        core, final_action, final_write = active_act_at_final_step()
        first_snapshot_captured = threading.Event()
        release_first_snapshot = threading.Event()
        original_snapshot = core._sink.snapshot
        call_count = 0

        def controlled_snapshot():
            nonlocal call_count
            snapshot = original_snapshot()
            call_count += 1
            if call_count == 1:
                first_snapshot_captured.set()
                self.assertTrue(release_first_snapshot.wait(timeout=1.0))
            return snapshot

        with tempfile.TemporaryDirectory() as temp_dir:
            socket_path = str(Path(temp_dir) / "control.sock")
            server = ResidentMotionControlServer(core, socket_path=socket_path)
            server.start()
            self.addCleanup(server.close)
            responses = []
            with mock.patch.object(core._sink, "snapshot", controlled_snapshot):
                status_thread = threading.Thread(
                    target=lambda: responses.append(
                        request_resident_motion_control(socket_path, "status")
                    )
                )
                status_thread.start()
                self.assertTrue(first_snapshot_captured.wait(timeout=1.0))
                completion_thread = threading.Thread(
                    target=lambda: core.observe_telemetry(
                        telemetry(
                            receive_ns=1_040_000_000,
                            command_seq=final_write.command_seq,
                            valid=True,
                            mode=ControlMode.MANUAL_ACTION,
                            action=final_action,
                        )
                    )
                )
                completion_thread.start()
                time.sleep(0.05)
                release_first_snapshot.set()
                status_thread.join(timeout=1.0)
                completion_thread.join(timeout=1.0)

        self.assertEqual(len(responses), 1)
        status = responses[0]["status"]
        impossible = (
            status["act_segment_complete"]
            and status["phase"] == "active"
            and status["active"]
            == {"source": "act_dig", "mode": "manual_action"}
            and status["target"] is None
        )
        self.assertFalse(impossible, status)


if __name__ == "__main__":
    unittest.main()
