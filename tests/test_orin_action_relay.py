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
            self.assertIn(b";0.00351;0;0;0\n", self.serial.writes[-1])
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
            self.assertTrue(self.serial.writes[0].endswith(b";0;0;0;0\n"))
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
            self.assertFalse(any(b";0.00351;" in item for item in self.serial.writes))
            self.assertTrue(self.serial.writes[-1].endswith(b";0;0;0;0\n"))
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
            while time.monotonic() < deadline and not self.serial.writes[-1].endswith(
                b";0;0;0;0\n"
            ):
                time.sleep(0.005)
            self.assertTrue(self.serial.writes[-1].endswith(b";0;0;0;0\n"))
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
            while time.monotonic() < deadline and not self.serial.writes[-1].endswith(
                b";0;0;0;0\n"
            ):
                time.sleep(0.005)
            self.assertTrue(self.serial.writes[-1].endswith(b";0;0;0;0\n"))
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
            self.assertFalse(any(b";0.00351;" in item for item in self.serial.writes))
            self.assertTrue(self.serial.writes[-1].endswith(b";0;0;0;0\n"))
        finally:
            relay.close()


if __name__ == "__main__":
    unittest.main()
