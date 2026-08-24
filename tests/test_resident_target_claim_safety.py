import json
import unittest

from edge_runtime.resident_motion import ControlMode, PolicyBinding, ZERO_ACTION
from edge_runtime.resident_sink import ResidentCommandSink, ResidentTelemetry


class _RecordingSerial:
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
    command_rx_seq: int,
    mode: ControlMode,
    control_enabled: bool = True,
) -> ResidentTelemetry:
    return ResidentTelemetry(
        receive_monotonic_ns=receive_ns,
        command_rx_seq=command_rx_seq,
        command_valid=True,
        command_timed_out=False,
        control_mode=mode,
        command_action=ZERO_ACTION,
        control_enabled=control_enabled,
        estop=False,
        sensor_valid=True,
        stm32_alive=True,
        fault_flags=0,
    )


class ResidentTargetClaimSafetyTest(unittest.TestCase):
    def test_target_zero_waits_for_full_safety_before_activating_rl(self) -> None:
        serial = _RecordingSerial()
        sink = ResidentCommandSink(serial, max_state_age_ms=200.0)
        sink.initialize(
            _telemetry(
                receive_ns=900_000_000,
                command_rx_seq=0,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        act = PolicyBinding("act_dig", ControlMode.MANUAL_ACTION)
        rl = PolicyBinding("rl_follow", ControlMode.VELOCITY_REFERENCE)

        sink.request_handoff(act, now_monotonic_ns=1_000_000_000)
        act_claim = json.loads(serial.writes[-1].decode("ascii"))
        sink.observe_telemetry(
            _telemetry(
                receive_ns=1_020_000_000,
                command_rx_seq=act_claim["command_seq"],
                mode=ControlMode.MANUAL_ACTION,
            )
        )

        handoff_generation = sink.request_handoff(
            rl,
            now_monotonic_ns=1_030_000_000,
        )
        terminal_zero = json.loads(serial.writes[-1].decode("ascii"))
        sink.observe_telemetry(
            _telemetry(
                receive_ns=1_050_000_000,
                command_rx_seq=terminal_zero["command_seq"],
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        target_zero = json.loads(serial.writes[-1].decode("ascii"))

        sink.observe_telemetry(
            _telemetry(
                receive_ns=1_060_000_000,
                command_rx_seq=target_zero["command_seq"],
                mode=ControlMode.VELOCITY_REFERENCE,
                control_enabled=False,
            )
        )

        waiting = sink.snapshot()
        self.assertEqual(waiting.generation, handoff_generation)
        self.assertEqual(waiting.phase.value, "target_zero_pending")
        self.assertIsNone(waiting.active_binding)
        self.assertEqual(waiting.target_binding, rl)

        sink.observe_telemetry(
            _telemetry(
                receive_ns=1_080_000_000,
                command_rx_seq=target_zero["command_seq"],
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )

        active = sink.snapshot()
        self.assertEqual(active.generation, handoff_generation)
        self.assertEqual(active.phase.value, "active")
        self.assertEqual(active.active_binding, rl)
        self.assertIsNone(active.target_binding)


if __name__ == "__main__":
    unittest.main()
