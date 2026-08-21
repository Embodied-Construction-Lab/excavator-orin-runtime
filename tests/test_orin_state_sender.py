import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import orin_state_sender as orin


LIVE_ROW = (
    "2560404,0.000,0.000,0.000,0.000,"
    "99.950,221.950,161.580,"
    "0.000,0.000,0.000,"
    "23.300,71.530,111.600,35.290,0.076,"
    "1,1,1"
)


class Stm32CsvSafetyFlagsTest(unittest.TestCase):
    def test_hardware_start_gate_waits_until_one_shot_file_arrives(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "rl.start"
            finished = threading.Event()

            def wait_for_gate():
                orin.wait_for_hardware_start_gate(gate, poll_interval_s=0.001)
                finished.set()

            worker = threading.Thread(target=wait_for_gate)
            worker.start()
            time.sleep(0.02)

            self.assertFalse(finished.is_set())
            gate.touch()
            worker.join(timeout=1.0)

            self.assertTrue(finished.is_set())
            self.assertFalse(gate.exists())

    def test_hardware_start_gate_requires_absolute_path(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            orin.wait_for_hardware_start_gate(
                Path("relative.start"), poll_interval_s=0.001
            )

    def test_cli_accepts_hardware_start_gate_without_changing_default(self):
        default_args = orin.parse_args([])
        gated_args = orin.parse_args(
            ["--hardware-start-gate", "/tmp/excavator-rl-control/test.start"]
        )

        self.assertIsNone(default_args.hardware_start_gate)
        self.assertEqual(
            gated_args.hardware_start_gate,
            Path("/tmp/excavator-rl-control/test.start"),
        )

    def test_cancelled_hardware_gate_opens_no_device_or_network_socket(self):
        args = SimpleNamespace(
            allowed_action_host=None,
            pc_host="192.168.50.1",
            edge_config=None,
            hardware_start_gate=Path(
                "/tmp/excavator-rl-control/hybrid_cancelled.start"
            ),
        )

        with (
            mock.patch.object(orin, "parse_args", return_value=args),
            mock.patch.object(orin.signal, "signal"),
            mock.patch.object(
                orin,
                "wait_for_hardware_start_gate",
                side_effect=KeyboardInterrupt,
            ),
            mock.patch.object(orin, "open_serial") as open_serial,
            mock.patch.object(orin, "open_action_socket") as open_action_socket,
            mock.patch.object(orin.socket, "socket") as open_udp_socket,
            mock.patch.object(orin, "send_udp_json") as send_machine_state,
        ):
            orin.main()

        open_serial.assert_not_called()
        open_action_socket.assert_not_called()
        open_udp_socket.assert_not_called()
        send_machine_state.assert_not_called()

    def test_sigterm_handler_enters_existing_keyboard_interrupt_cleanup_path(self):
        with self.assertRaises(KeyboardInterrupt):
            orin._raise_keyboard_interrupt_on_termination(None, None)

    def test_unified_firmware_defaults_match_field_serial_link(self):
        self.assertEqual(orin.DEFAULT_SERIAL_PORT, "/dev/ttyTHS1")
        self.assertEqual(orin.DEFAULT_BAUDRATE, 460800)

    def test_unified_v2_row_maps_state_safety_and_command_sequence(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version="stm32_control_telemetry.v2",
            control_stamp_ms="1234",
            command_rx_seq="41",
            command_valid="1",
            control_enabled="1",
            boom_pos_mm="100.0",
            stick_pos_mm="120.0",
            bucket_pos_mm="80.0",
            boom_vel_mmps="2.0",
            stick_vel_mmps="-3.0",
            bucket_vel_mmps="4.0",
            boom_angle_deg="10.0",
            arm_angle_deg="20.0",
            bucket_angle_deg="30.0",
            swing_angle_deg="40.0",
            swing_vel_degps="5.0",
            rs485_ok="1",
            dwj_ok="1",
            imu_ok="1",
        )
        row = ",".join(values[field] for field in orin.STM32_V2_FIELDS)

        state = orin.parse_stm32_csv_line(row)

        self.assertIsNotNone(state)
        self.assertEqual(state.command_rx_seq, 41)
        self.assertTrue(state.command_received)
        self.assertTrue(state.stm32_control_enabled)
        self.assertAlmostEqual(state.s_boom, 0.1)
        self.assertAlmostEqual(state.v_stick, -0.003)
        self.assertAlmostEqual(state.yaw, 0.6981317008)

    def test_unified_v2_rejects_corruption_in_any_telemetry_field(self):
        values = {field: "0" for field in orin.STM32_V2_FIELDS}
        values.update(
            schema_version="stm32_control_telemetry.v2",
            pid_out_boom="nan",
        )
        row = ",".join(values[field] for field in orin.STM32_V2_FIELDS)

        with self.assertRaisesRegex(ValueError, "pid_out_boom"):
            orin.parse_stm32_csv_line(row)

    def test_live_19_field_row_maps_all_hardware_validity_flags(self):
        state = orin.parse_stm32_csv_line(LIVE_ROW)

        self.assertIsNotNone(state)
        self.assertTrue(state.rs485_ok)
        self.assertTrue(state.adc_ok)
        self.assertTrue(state.imu_ok)
        packet = orin.build_machine_state_packet(
            state=state,
            seq=1,
            machine_id="scale_excavator_v1",
            control_enabled=False,
            estop=False,
            include_raw=True,
            last_receive_monotonic_s=time.monotonic(),
        )
        self.assertTrue(packet["safety"]["sensor_valid"])
        self.assertEqual(packet["safety"]["fault_flags"], [])
        self.assertTrue(packet["raw_sensor"]["rs485_ok"])
        self.assertTrue(packet["raw_sensor"]["adc_ok"])
        self.assertTrue(packet["raw_sensor"]["imu_ok"])

    def test_each_zero_hardware_flag_fails_closed_with_specific_fault(self):
        cases = {
            16: "rs485_invalid",
            17: "adc_invalid",
            18: "imu_invalid",
        }
        for field_index, expected_fault in cases.items():
            with self.subTest(expected_fault=expected_fault):
                fields = LIVE_ROW.split(",")
                fields[field_index] = "0"
                state = orin.parse_stm32_csv_line(",".join(fields))
                packet = orin.build_machine_state_packet(
                    state=state,
                    seq=2,
                    machine_id="scale_excavator_v1",
                    control_enabled=False,
                    estop=False,
                    include_raw=False,
                    last_receive_monotonic_s=time.monotonic(),
                )

                self.assertFalse(packet["safety"]["sensor_valid"])
                self.assertIn(expected_fault, packet["safety"]["fault_flags"])

    def test_partial_or_non_boolean_hardware_flags_are_rejected(self):
        fields = LIVE_ROW.split(",")
        with self.assertRaisesRegex(ValueError, "validity field count"):
            orin.parse_stm32_csv_line(",".join(fields[:18]))

        fields[16] = "2"
        with self.assertRaisesRegex(ValueError, "rs485_ok"):
            orin.parse_stm32_csv_line(",".join(fields))

    def test_legacy_16_field_row_remains_supported(self):
        state = orin.parse_stm32_csv_line(",".join(LIVE_ROW.split(",")[:16]))

        self.assertTrue(state.rs485_ok)
        self.assertTrue(state.adc_ok)
        self.assertTrue(state.imu_ok)


if __name__ == "__main__":
    unittest.main()
