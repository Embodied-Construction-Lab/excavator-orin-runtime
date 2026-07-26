import time
import unittest

import orin_state_sender as orin


LIVE_ROW = (
    "2560404,0.000,0.000,0.000,0.000,"
    "99.950,221.950,161.580,"
    "0.000,0.000,0.000,"
    "23.300,71.530,111.600,35.290,0.076,"
    "1,1,1"
)


class Stm32CsvSafetyFlagsTest(unittest.TestCase):
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
