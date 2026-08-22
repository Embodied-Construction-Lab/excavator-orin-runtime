import json
import math
import unittest

from edge_runtime.resident_commands import Stm32ResidentCommandEncoder
from edge_runtime.resident_motion import ControlMode, ZERO_ACTION


class Stm32ResidentCommandEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = Stm32ResidentCommandEncoder()

    def decode(self, payload: bytes) -> dict:
        self.assertTrue(payload.endswith(b"\n"))
        return json.loads(payload.decode("ascii"))

    def test_synchronize_resumes_the_single_shared_uint32_sequence(self) -> None:
        self.assertEqual(
            self.encoder.synchronize(command_rx_seq=41, command_received=True),
            42,
        )
        self.assertEqual(self.encoder.next_sequence, 42)

        fresh = Stm32ResidentCommandEncoder()
        self.assertEqual(
            fresh.synchronize(command_rx_seq=3_000, command_received=False),
            0,
        )

    def test_manual_action_maps_canonical_axes_without_sign_or_scale_changes(self) -> None:
        self.encoder.synchronize(command_rx_seq=9, command_received=True)
        payload = self.decode(
            self.encoder.encode(
                mode=ControlMode.MANUAL_ACTION,
                action=(0.11, -0.22, 0.33, -0.44),
                monotonic_ns=12_345_678_999,
            )
        )

        self.assertEqual(payload["schema_version"], "stm32_manual_command.v1")
        self.assertEqual(
            {name: payload[name] for name in ("X1", "Y1", "Z1", "X2", "Y2", "Z2")},
            {
                "X1": -0.44,
                "Y1": -0.22,
                "Z1": 0.0,
                "X2": 0.33,
                "Y2": 0.11,
                "Z2": 0.0,
            },
        )
        self.assertEqual(payload["command_seq"], 10)
        self.assertEqual(payload["command_source_stamp_ms"], 12_345)

    def test_velocity_reference_preserves_physical_units_and_order(self) -> None:
        payload = self.decode(
            self.encoder.encode(
                mode=ControlMode.VELOCITY_REFERENCE,
                action=(0.025, -0.03, 0.04, -0.5),
                monotonic_ns=987_654_321,
            )
        )

        self.assertEqual(payload["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(
            {
                name: payload[name]
                for name in ("boom_mps", "stick_mps", "bucket_mps", "swing_radps")
            },
            {
                "boom_mps": 0.025,
                "stick_mps": -0.03,
                "bucket_mps": 0.04,
                "swing_radps": -0.5,
            },
        )
        self.assertEqual(payload["command_seq"], 0)

    def test_sequence_is_continuous_across_mode_switch_and_wrap(self) -> None:
        self.encoder.synchronize(command_rx_seq=0xFFFFFFFE, command_received=True)

        velocity_zero = self.decode(
            self.encoder.encode(
                mode=ControlMode.VELOCITY_REFERENCE,
                action=ZERO_ACTION,
                monotonic_ns=1_000_000,
            )
        )
        manual_zero = self.decode(
            self.encoder.encode(
                mode=ControlMode.MANUAL_ACTION,
                action=ZERO_ACTION,
                monotonic_ns=2_000_000,
            )
        )
        manual_motion = self.decode(
            self.encoder.encode(
                mode=ControlMode.MANUAL_ACTION,
                action=(0.1, 0.0, 0.0, 0.0),
                monotonic_ns=3_000_000,
            )
        )

        self.assertEqual(velocity_zero["command_seq"], 0xFFFFFFFF)
        self.assertEqual(manual_zero["command_seq"], 0)
        self.assertEqual(manual_motion["command_seq"], 1)
        self.assertEqual(self.encoder.next_sequence, 2)

    def test_rejects_ambiguous_or_invalid_wire_values(self) -> None:
        invalid = (
            (ControlMode.MANUAL_ACTION, (1.01, 0.0, 0.0, 0.0), 1),
            (ControlMode.MANUAL_ACTION, (math.nan, 0.0, 0.0, 0.0), 1),
            (ControlMode.MANUAL_ACTION, (True, 0.0, 0.0, 0.0), 1),
            (ControlMode.VELOCITY_REFERENCE, (math.inf, 0.0, 0.0, 0.0), 1),
            (ControlMode.VELOCITY_REFERENCE, (0.0, 0.0, 0.0), 1),
            (ControlMode.VELOCITY_REFERENCE, ZERO_ACTION, -1),
        )
        for mode, action, monotonic_ns in invalid:
            with self.subTest(mode=mode, action=action, monotonic_ns=monotonic_ns):
                with self.assertRaises(ValueError):
                    self.encoder.encode(
                        mode=mode,
                        action=action,
                        monotonic_ns=monotonic_ns,
                    )

    def test_synchronize_rejects_non_uint32_values(self) -> None:
        for value in (-1, 0x1_0000_0000, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.encoder.synchronize(
                        command_rx_seq=value,
                        command_received=True,
                    )


if __name__ == "__main__":
    unittest.main()
