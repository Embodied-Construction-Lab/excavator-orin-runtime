import json
import math
import unittest

from edge_runtime.resident_state import (
    ACT_STATE_NAMES,
    MAX_RESIDENT_STATE_BYTES,
    RESIDENT_STATE_SCHEMA_VERSION,
    UINT32_MAX,
    UINT64_MAX,
    ResidentActState,
    decode_resident_state,
    encode_resident_state,
)


class ResidentActStateProtocolTest(unittest.TestCase):
    def state(self) -> ResidentActState:
        return ResidentActState(
            state=(
                0.101,
                0.202,
                0.303,
                0.011,
                -0.022,
                0.033,
                0.41,
                0.52,
                0.63,
                -0.74,
                0.085,
            ),
            receive_monotonic_ns=2_000_000_000,
            state_monotonic_ns=1_999_000_000,
            control_seq=41,
            sensor_seq=20,
            sensor_is_new=True,
            control_enabled=True,
            estop=False,
            rs485_ok=True,
            dwj_ok=True,
            imu_ok=True,
            sensor_valid=True,
            stm32_alive=True,
            fault_flags=0,
            control_generation=7,
        )

    def test_round_trip_preserves_named_act_state_and_control_evidence(
        self,
    ) -> None:
        original = self.state()

        encoded = encode_resident_state(original)
        decoded = decode_resident_state(encoded)

        self.assertEqual(decoded, original)
        self.assertLessEqual(len(encoded), MAX_RESIDENT_STATE_BYTES)
        payload = json.loads(encoded)
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "state_names",
                "state",
                "receive_monotonic_ns",
                "state_monotonic_ns",
                "control_seq",
                "sensor_seq",
                "sensor_is_new",
                "control_enabled",
                "estop",
                "rs485_ok",
                "dwj_ok",
                "imu_ok",
                "sensor_valid",
                "stm32_alive",
                "fault_flags",
                "control_generation",
            },
        )
        self.assertEqual(payload["schema_version"], RESIDENT_STATE_SCHEMA_VERSION)
        self.assertEqual(payload["state_names"], list(ACT_STATE_NAMES))
        self.assertEqual(
            payload["state_names"],
            [
                "boom_pos_m",
                "stick_pos_m",
                "bucket_pos_m",
                "boom_vel_mps",
                "stick_vel_mps",
                "bucket_vel_mps",
                "boom_angle_rad",
                "arm_angle_rad",
                "bucket_angle_rad",
                "swing_angle_rad",
                "swing_vel_radps",
            ],
        )

    def test_integer_ranges_are_inclusive_and_sensor_heartbeat_is_valid(self) -> None:
        values = dict(self.state().__dict__)
        minimum = ResidentActState(
            **{
                **values,
                "state": (0.0,) * len(ACT_STATE_NAMES),
                "receive_monotonic_ns": 0,
                "state_monotonic_ns": 0,
                "control_seq": 0,
                "sensor_seq": 0,
                "sensor_is_new": False,
                "fault_flags": 0,
                "control_generation": 0,
            }
        )
        boundary = ResidentActState(
            **{
                **values,
                "receive_monotonic_ns": UINT64_MAX,
                "state_monotonic_ns": UINT64_MAX,
                "control_seq": UINT32_MAX,
                "sensor_seq": UINT32_MAX,
                "sensor_is_new": False,
                "fault_flags": UINT32_MAX,
                "control_generation": UINT64_MAX,
            }
        )

        self.assertEqual(
            decode_resident_state(encode_resident_state(minimum)),
            minimum,
        )
        self.assertEqual(
            decode_resident_state(encode_resident_state(boundary)),
            boundary,
        )

    def test_round_trip_preserves_unsafe_and_heartbeat_safety_evidence(self) -> None:
        values = dict(self.state().__dict__)
        unsafe = ResidentActState(
            **{
                **values,
                "sensor_is_new": False,
                "control_enabled": False,
                "estop": True,
                "rs485_ok": False,
                "dwj_ok": False,
                "imu_ok": False,
                "sensor_valid": False,
                "stm32_alive": False,
                "fault_flags": 0xA5,
            }
        )

        encoded = encode_resident_state(unsafe)
        payload = json.loads(encoded)

        self.assertEqual(decode_resident_state(encoded), unsafe)
        self.assertEqual(
            {
                field: payload[field]
                for field in (
                    "sensor_is_new",
                    "control_enabled",
                    "estop",
                    "rs485_ok",
                    "dwj_ok",
                    "imu_ok",
                    "sensor_valid",
                    "stm32_alive",
                    "fault_flags",
                )
            },
            {
                "sensor_is_new": False,
                "control_enabled": False,
                "estop": True,
                "rs485_ok": False,
                "dwj_ok": False,
                "imu_ok": False,
                "sensor_valid": False,
                "stm32_alive": False,
                "fault_flags": 0xA5,
            },
        )

    def test_decoder_requires_the_exact_versioned_field_set(self) -> None:
        payload = json.loads(encode_resident_state(self.state()))
        missing = dict(payload)
        missing.pop("control_generation")
        extra = {**payload, "units": "implicit"}
        bad_version = {**payload, "schema_version": "resident_act_state.v0"}

        for mutation in (missing, extra, bad_version):
            with self.subTest(fields=mutation.keys()):
                with self.assertRaises(ValueError):
                    decode_resident_state(json.dumps(mutation).encode("utf-8"))

    def test_decoder_rejects_wrong_state_names_order_and_dimension(self) -> None:
        payload = json.loads(encode_resident_state(self.state()))
        reordered = dict(payload)
        reordered["state_names"] = [
            "swing_angle_rad",
            *payload["state_names"][:-2],
            "swing_vel_radps",
        ]
        renamed = dict(payload)
        renamed["state_names"] = [
            "stick_angle_rad" if name == "arm_angle_rad" else name
            for name in payload["state_names"]
        ]
        short_state = {**payload, "state": payload["state"][:-1]}

        for mutation in (reordered, renamed, short_state):
            with self.subTest(state_names=mutation["state_names"]):
                with self.assertRaises(ValueError):
                    decode_resident_state(json.dumps(mutation).encode("utf-8"))

    def test_decoder_rejects_nonfinite_or_non_numeric_state_values(self) -> None:
        payload = json.loads(encode_resident_state(self.state()))
        for invalid in (
            math.nan,
            math.inf,
            -math.inf,
            10**400,
            True,
            "0.101",
            None,
        ):
            mutation = dict(payload)
            mutation["state"] = [invalid, *payload["state"][1:]]
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    decode_resident_state(json.dumps(mutation).encode("utf-8"))

    def test_decoder_rejects_wrong_scalar_types_and_integer_ranges(self) -> None:
        payload = json.loads(encode_resident_state(self.state()))
        mutations = (
            {**payload, "receive_monotonic_ns": -1},
            {**payload, "state_monotonic_ns": 1.0},
            {**payload, "control_seq": True},
            {**payload, "control_seq": 0x1_0000_0000},
            {**payload, "sensor_seq": -1},
            {**payload, "fault_flags": 0x1_0000_0000},
            {**payload, "control_generation": True},
            {**payload, "control_generation": 0x1_0000_0000_0000_0000},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    decode_resident_state(json.dumps(mutation).encode("utf-8"))

        for field in (
            "sensor_is_new",
            "control_enabled",
            "estop",
            "rs485_ok",
            "dwj_ok",
            "imu_ok",
            "sensor_valid",
            "stm32_alive",
        ):
            mutation = {**payload, field: 1}
            with self.subTest(bool_field=field):
                with self.assertRaises(ValueError):
                    decode_resident_state(json.dumps(mutation).encode("utf-8"))

    def test_decoder_rejects_a_state_timestamp_after_receive(self) -> None:
        payload = json.loads(encode_resident_state(self.state()))
        payload["state_monotonic_ns"] = payload["receive_monotonic_ns"] + 1

        with self.assertRaisesRegex(ValueError, "state_monotonic_ns"):
            decode_resident_state(json.dumps(payload).encode("utf-8"))

    def test_decoder_rejects_invalid_or_oversize_payloads(self) -> None:
        invalid_payloads = (
            b"",
            b"\xff",
            b"[]",
            b"null",
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload[:16]):
                with self.assertRaises(ValueError):
                    decode_resident_state(payload)

        encoded = encode_resident_state(self.state())
        exact_limit = encoded + b" " * (MAX_RESIDENT_STATE_BYTES - len(encoded))
        self.assertEqual(decode_resident_state(exact_limit), self.state())
        with self.assertRaises(ValueError):
            decode_resident_state(exact_limit + b" ")

        for payload in (
            encoded.decode("ascii"),
            bytearray(encoded),
            memoryview(encoded),
        ):
            with self.subTest(payload_type=type(payload)):
                with self.assertRaises(ValueError):
                    decode_resident_state(payload)  # type: ignore[arg-type]

    def test_decoder_rejects_duplicate_json_fields(self) -> None:
        encoded = encode_resident_state(self.state())
        duplicate = encoded.replace(
            b"{",
            b'{"schema_version":"resident_act_state.v1",',
            1,
        )

        with self.assertRaisesRegex(ValueError, "strict finite JSON"):
            decode_resident_state(duplicate)

    def test_typed_frame_and_encoder_reject_invalid_python_inputs(self) -> None:
        values = dict(self.state().__dict__)
        invalid_frames = (
            {**values, "state": list(values["state"])},
            {**values, "state": (True, *values["state"][1:])},
            {**values, "state": (*values["state"][:-1], math.nan)},
            {**values, "sensor_seq": True},
            {**values, "control_enabled": 1},
            {**values, "state_monotonic_ns": values["receive_monotonic_ns"] + 1},
        )
        for invalid in invalid_frames:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ResidentActState(**invalid)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            encode_resident_state(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
