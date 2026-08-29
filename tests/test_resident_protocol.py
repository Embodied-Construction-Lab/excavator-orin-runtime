import json
import math
import unittest

from edge_runtime.resident_motion import ControlMode, MotionCandidate
from edge_runtime.resident_protocol import (
    CANDIDATE_SCHEMA_VERSION,
    decode_motion_candidate,
    encode_motion_candidate,
)


class ResidentPolicyProtocolTest(unittest.TestCase):
    def candidate(self, *, mode=ControlMode.MANUAL_ACTION) -> MotionCandidate:
        return MotionCandidate(
            source="act_dig" if mode is ControlMode.MANUAL_ACTION else "rl_follow",
            generation=7,
            mode=mode,
            action=(0.1, -0.2, 0.3, -0.4),
            created_monotonic_ns=10_000,
            valid_until_monotonic_ns=20_000,
        )

    def test_manual_candidate_round_trip_preserves_a_new_execution_chunk(self) -> None:
        chunk = tuple(
            (0.01 * index, -0.01 * index, 0.0, 0.0)
            for index in range(10)
        )
        candidate = MotionCandidate(
            **{**self.candidate().__dict__, "action": chunk[0], "action_chunk": chunk}
        )

        encoded = encode_motion_candidate(candidate)

        self.assertEqual(decode_motion_candidate(encoded), candidate)
        payload = json.loads(encoded)
        self.assertEqual(payload["schema_version"], "resident_policy_candidate.v2")
        self.assertEqual(payload["action_chunk"], [list(action) for action in chunk])

    def test_action_chunk_is_exactly_ten_normalized_manual_actions(self) -> None:
        base = self.candidate()
        for chunk in (
            ((0.0, 0.0, 0.0, 0.0),) * 9,
            ((0.0, 0.0, 0.0, 0.0),) * 11,
            ((0.0, 0.0, 0.0, 1.01),) * 10,
        ):
            with self.subTest(chunk=chunk):
                with self.assertRaises(ValueError):
                    MotionCandidate(**{**base.__dict__, "action_chunk": chunk})

    def test_round_trip_preserves_typed_action_without_scale_or_reorder(self) -> None:
        for mode in (ControlMode.MANUAL_ACTION, ControlMode.VELOCITY_REFERENCE):
            with self.subTest(mode=mode):
                original = self.candidate(mode=mode)
                encoded = encode_motion_candidate(original)
                self.assertLessEqual(len(encoded), 4096)
                decoded = decode_motion_candidate(encoded)
                self.assertEqual(decoded, original)

                payload = json.loads(encoded)
                self.assertEqual(payload["schema_version"], CANDIDATE_SCHEMA_VERSION)
                self.assertEqual(
                    payload["action_order"],
                    ["boom", "stick", "bucket", "swing"],
                )
                self.assertEqual(payload["mode"], mode.value)
                self.assertEqual(payload["control_generation"], 7)
                self.assertNotIn("generation", payload)

    def test_decoder_requires_the_exact_versioned_field_set(self) -> None:
        payload = json.loads(encode_motion_candidate(self.candidate()))
        mutations = []
        missing = dict(payload)
        missing.pop("control_generation")
        mutations.append(missing)
        extra = dict(payload)
        extra["units"] = "ambiguous"
        mutations.append(extra)
        bad_schema = dict(payload)
        bad_schema["schema_version"] = "resident_policy_candidate.v0"
        mutations.append(bad_schema)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    decode_motion_candidate(json.dumps(mutation).encode("utf-8"))

    def test_decoder_rejects_noncanonical_action_order(self) -> None:
        payload = json.loads(encode_motion_candidate(self.candidate()))
        payload["action_order"] = ["swing", "boom", "stick", "bucket"]
        with self.assertRaisesRegex(ValueError, "action_order"):
            decode_motion_candidate(json.dumps(payload).encode("utf-8"))

    def test_decoder_rejects_invalid_scalar_types_and_nonfinite_values(self) -> None:
        base = json.loads(encode_motion_candidate(self.candidate()))
        mutations = (
            {**base, "control_generation": True},
            {**base, "control_generation": 0x1_0000_0000_0000_0000},
            {**base, "created_monotonic_ns": -1},
            {**base, "mode": "manual"},
            {**base, "source": "   "},
            {**base, "action": [0.0, 0.0, 0.0, math.inf]},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                encoded = json.dumps(mutation).encode("utf-8")
                with self.assertRaises(ValueError):
                    decode_motion_candidate(encoded)

    def test_decoder_rejects_duplicate_json_fields(self) -> None:
        encoded = encode_motion_candidate(self.candidate())
        duplicate = encoded.replace(
            b"{",
            b'{"control_generation":7,',
            1,
        )

        with self.assertRaisesRegex(ValueError, "strict finite JSON"):
            decode_motion_candidate(duplicate)

    def test_decoder_rejects_oversize_non_utf8_and_non_object_payloads(self) -> None:
        invalid = (
            b"\xff",
            b"[]",
            b"{" + b" " * 4096 + b"}",
        )
        for payload in invalid:
            with self.subTest(payload=payload[:16]):
                with self.assertRaises(ValueError):
                    decode_motion_candidate(payload)

    def test_typed_candidate_uses_uint64_control_and_monotonic_ranges(self) -> None:
        values = dict(self.candidate().__dict__)
        for field in (
            "generation",
            "created_monotonic_ns",
            "valid_until_monotonic_ns",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    MotionCandidate(
                        **{**values, field: 0x1_0000_0000_0000_0000}
                    )


if __name__ == "__main__":
    unittest.main()
