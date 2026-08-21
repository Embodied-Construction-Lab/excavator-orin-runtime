import importlib.util
import json
import socket
import sys
import threading
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "orin_state_sender.py"
SPEC = importlib.util.spec_from_file_location("orin_state_sender", MODULE_PATH)
orin = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = orin
SPEC.loader.exec_module(orin)


class RecordingSerial:
    def __init__(self):
        self.writes = []
        self.written = threading.Event()

    def write(self, payload):
        self.writes.append(payload)
        self.written.set()
        return len(payload)

    def flush(self):
        return None


def action_packet(sequence, action, valid_for_ms=100, stamp_ms=None):
    return json.dumps(
        {
            "type": "policy_action",
            "schema_version": "1.0",
            "seq": sequence,
            "stamp_ms": orin.now_ms() if stamp_ms is None else stamp_ms,
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": action,
            "action_type": "normalized_velocity_command",
            "valid_for_ms": valid_for_ms,
        }
    ).encode("utf-8")


def stm32_command(payload):
    return json.loads(payload.decode("ascii"))


def is_zero_command(payload):
    command = stm32_command(payload)
    return all(
        command[name] == 0.0
        for name in ("boom_mps", "stick_mps", "bucket_mps", "swing_radps")
    )


class ActionRelayTest(unittest.TestCase):
    def setUp(self):
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.setblocking(False)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serial = RecordingSerial()

    def tearDown(self):
        self.sender.close()
        self.receiver.close()

    def test_action_source_host_can_be_configured_independently_from_pc_state_host(self):
        args = orin.parse_args(
            [
                "--pc-host",
                "192.0.2.10",
                "--allowed-action-host",
                "127.0.0.1",
            ]
        )

        self.assertEqual(args.pc_host, "192.0.2.10")
        self.assertEqual(args.allowed_action_host, "127.0.0.1")

    def test_velocity_encoder_resumes_stm32_sequence_and_preserves_physical_units(self):
        encoder = orin.Stm32VelocityCommandEncoder()
        encoder.synchronize(command_rx_seq=41, command_received=True)
        command = orin.DataCommand(
            t_s=1.234567,
            boom_v_ref_mps=0.12,
            stick_v_ref_mps=-0.08,
            bucket_v_ref_mps=0.04,
            swing_v_ref_radps=-0.15,
        )

        payload = json.loads(encoder.encode(command))

        self.assertEqual(payload["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(payload["command_seq"], 42)
        self.assertEqual(payload["command_source_stamp_ms"], 1234)
        self.assertEqual(
            [
                payload["boom_mps"],
                payload["stick_mps"],
                payload["bucket_mps"],
                payload["swing_radps"],
            ],
            [0.12, -0.08, 0.04, -0.15],
        )

    def test_relay_zero_mode_claim_continues_from_stm32_ack_sequence(self):
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        state = orin.Stm32State(
            *(0 for _ in range(16)),
            command_rx_seq=41,
            command_received=True,
        )
        relay.synchronize_command_sequence(state)
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        try:
            self.sender.sendto(
                action_packet(1, [0.01, 0.0, 0.0, 0.0]),
                self.receiver.getsockname(),
            )
            deadline = time.monotonic() + 0.1
            while len(self.serial.writes) < 2 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(stm32_command(self.serial.writes[0])["command_seq"], 42)
            self.assertTrue(is_zero_command(self.serial.writes[0]))
            self.assertEqual(stm32_command(self.serial.writes[1])["command_seq"], 43)
        finally:
            relay.close()

    def test_receives_action_without_waiting_for_another_state_frame(self):
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        started = time.monotonic()
        try:
            self.sender.sendto(
                action_packet(1, [0.00351, 0.0, 0.0, 0.0]),
                self.receiver.getsockname(),
            )
            self.assertTrue(self.serial.written.wait(0.08))
            self.assertLess(time.monotonic() - started, 0.08)
            deadline = time.monotonic() + 0.08
            while len(self.serial.writes) < 2 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertGreaterEqual(len(self.serial.writes), 2)
            self.assertTrue(is_zero_command(self.serial.writes[-2]))
            self.assertEqual(stm32_command(self.serial.writes[-2])["command_seq"], 0)
            command = stm32_command(self.serial.writes[-1])
            self.assertEqual(command["schema_version"], "stm32_velocity_command.v1")
            self.assertEqual(command["command_seq"], 1)
            self.assertEqual(command["boom_mps"], 0.00351)
        finally:
            relay.close()

    def test_relays_finite_action_values_without_orin_range_check(self):
        packet = json.loads(action_packet(1, [12.5, -9.25, 3.75, -7.5]))

        command, reject_reasons = orin.policy_action_to_data_command(
            packet,
            command_time_s=1.0,
            receive_wall_ms=packet["stamp_ms"],
            control_enabled=True,
            estop=False,
            sensor_valid=True,
            stm32_alive=True,
        )

        self.assertEqual(reject_reasons, [])
        self.assertEqual(
            (
                command.boom_v_ref_mps,
                command.stick_v_ref_mps,
                command.bucket_v_ref_mps,
                command.swing_v_ref_radps,
            ),
            (12.5, -9.25, 3.75, -7.5),
        )

    def test_policy_action_console_log_labels_each_physical_command(self):
        command = orin.DataCommand(
            t_s=2080.229,
            boom_v_ref_mps=-0.0146955703,
            stick_v_ref_mps=-0.0329888032,
            bucket_v_ref_mps=0.0,
            swing_v_ref_radps=0.6,
        )

        message = orin.format_policy_action_tx_log(sequence=379, command=command)

        self.assertEqual(
            message,
            "STM32 TX policy_action seq=379 t_ms:2080229 "
            "boom:-0.0146955703m/s stick:-0.0329888032m/s "
            "bucket:0m/s swing:0.6rad/s",
        )

    def test_discards_older_queued_actions_and_applies_only_latest(self):
        destination = self.receiver.getsockname()
        self.sender.sendto(action_packet(1, [0.00351, 0.0, 0.0, 0.0]), destination)
        self.sender.sendto(action_packet(2, [0.0, 0.0, 0.0, 0.0]), destination)
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        try:
            self.assertTrue(self.serial.written.wait(0.08))
            time.sleep(0.02)
            self.assertEqual(self.serial.writes, [self.serial.writes[0]])
            self.assertTrue(is_zero_command(self.serial.writes[0]))
        finally:
            relay.close()

    def test_udp_reordering_cannot_apply_old_nonzero_after_new_zero(self):
        destination = self.receiver.getsockname()
        self.sender.sendto(action_packet(2, [0.0, 0.0, 0.0, 0.0]), destination)
        self.sender.sendto(action_packet(1, [0.00351, 0.0, 0.0, 0.0]), destination)
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        try:
            self.assertTrue(self.serial.written.wait(0.08))
            time.sleep(0.02)
            self.assertFalse(
                any(stm32_command(item)["boom_mps"] == 0.00351 for item in self.serial.writes)
            )
            self.assertTrue(is_zero_command(self.serial.writes[-1]))
        finally:
            relay.close()

    def test_validity_deadline_actively_writes_zero(self):
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        try:
            self.sender.sendto(
                action_packet(1, [0.00351, 0.0, 0.0, 0.0], valid_for_ms=50),
                self.receiver.getsockname(),
            )
            self.assertTrue(self.serial.written.wait(0.08))
            deadline = time.monotonic() + 0.12
            while time.monotonic() < deadline and not is_zero_command(
                self.serial.writes[-1]
            ):
                time.sleep(0.005)
            self.assertTrue(is_zero_command(self.serial.writes[-1]))
        finally:
            relay.close()

    def test_safety_transition_actively_writes_zero(self):
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        try:
            self.sender.sendto(
                action_packet(1, [0.00351, 0.0, 0.0, 0.0], valid_for_ms=500),
                self.receiver.getsockname(),
            )
            self.assertTrue(self.serial.written.wait(0.08))
            relay.update_safety(sensor_valid=False, stm32_alive=True)
            deadline = time.monotonic() + 0.08
            while time.monotonic() < deadline and not is_zero_command(
                self.serial.writes[-1]
            ):
                time.sleep(0.005)
            self.assertTrue(is_zero_command(self.serial.writes[-1]))
        finally:
            relay.close()

    def test_future_timestamp_is_rejected_with_zero(self):
        relay = orin.ActionRelay(
            action_sock=self.receiver,
            ser=self.serial,
            allowed_action_host="127.0.0.1",
            control_enabled=True,
            estop=False,
            poll_interval_s=0.005,
        )
        relay.update_safety(sensor_valid=True, stm32_alive=True)
        relay.start()
        try:
            self.sender.sendto(
                action_packet(
                    1,
                    [0.00351, 0.0, 0.0, 0.0],
                    stamp_ms=orin.now_ms() + 3_600_000,
                ),
                self.receiver.getsockname(),
            )
            self.assertTrue(self.serial.written.wait(0.08))
            self.assertFalse(
                any(stm32_command(item)["boom_mps"] == 0.00351 for item in self.serial.writes)
            )
            self.assertTrue(is_zero_command(self.serial.writes[-1]))
        finally:
            relay.close()


if __name__ == "__main__":
    unittest.main()
