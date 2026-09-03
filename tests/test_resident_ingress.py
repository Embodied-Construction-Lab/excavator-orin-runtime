import json
import tempfile
import unittest
from pathlib import Path

from edge_runtime.control import EdgeControlRunner
from edge_runtime.follow import EdgeFollowStep
from edge_runtime.resident_ingress import (
    ResidentPolicyCandidateAdapter,
    ResidentVelocityActionAdapter,
)
from edge_runtime.resident_motion import (
    ControlMode,
    MotionCandidate,
    PolicyBinding,
    ZERO_ACTION,
)
from edge_runtime.resident_protocol import encode_motion_candidate
from edge_runtime.resident_sink import ResidentCommandSink, ResidentTelemetry


class RecordingSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None


class StubFollowRuntime:
    def step(self, machine_state, *, now_s):
        return EdgeFollowStep(
            source_seq=machine_state["seq"],
            source_stamp_ms=machine_state["stamp_ms"],
            waypoint_index=0,
            completed=False,
            bucket_tip_ros_m=(0.1, 0.2, 0.3),
            bucket_pitch_rad=0.4,
            observation=tuple(0.0 for _ in range(38)),
            normalized_action=(0.1, -0.2, 0.3, -0.4),
            physical_action=(0.025, -0.03, 0.04, -0.5),
            commanded_normalized_action=(0.1, -0.2, 0.3, -0.4),
            episode_progress=0.1,
            waypoint_distance_m=0.4,
            trajectory_controller_backend="onnx_rl",
            reference_waypoint_ros_m=(0.8, -0.1, 0.0),
        )


def telemetry(
    *, receive_ns: int, command_seq: int, valid: bool, mode: ControlMode | None
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


def policy_action(
    *,
    sequence: int,
    stamp_ms: int,
    action=(0.025, -0.03, 0.04, -0.5),
    valid_for_ms: int = 100,
) -> bytes:
    return json.dumps(
        {
            "type": "policy_action",
            "schema_version": "1.0",
            "seq": sequence,
            "stamp_ms": stamp_ms,
            "action_order": ["boom", "stick", "bucket", "swing"],
            "action": list(action),
            "action_type": "normalized_velocity_command",
            "valid_for_ms": valid_for_ms,
        },
        separators=(",", ":"),
    ).encode("utf-8")


class ResidentVelocityActionAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = RecordingSerial()
        self.sink = ResidentCommandSink(self.serial, max_state_age_ms=200.0)
        self.sink.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        self.adapter = ResidentVelocityActionAdapter(
            self.sink,
            source="rl_follow",
            wall_time_ms=lambda: 10_000,
            monotonic_ns=lambda: 1_030_000_000,
        )

    def packets(self) -> list[dict]:
        return [json.loads(payload.decode("ascii")) for payload in self.serial.writes]

    def activate(self) -> int:
        generation = self.adapter.begin_activation(now_monotonic_ns=1_000_000_000)
        claim = self.packets()[-1]
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_020_000_000,
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.VELOCITY_REFERENCE,
            )
        )
        return generation

    def test_existing_edge_control_packet_enters_resident_velocity_mode_unchanged(self) -> None:
        self.activate()
        result = self.adapter.send(
            policy_action(sequence=7, stamp_ms=9_980)
        )

        self.assertTrue(result.accepted)
        packet = self.packets()[-1]
        self.assertEqual(packet["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(
            [
                packet["boom_mps"],
                packet["stick_mps"],
                packet["bucket_mps"],
                packet["swing_radps"],
            ],
            [0.025, -0.03, 0.04, -0.5],
        )

    def test_existing_edge_control_runner_can_use_resident_adapter_directly(self) -> None:
        self.activate()
        with tempfile.TemporaryDirectory() as directory:
            runner = EdgeControlRunner(
                runtime=StubFollowRuntime(),
                action_sink=self.adapter,
                audit_path=Path(directory) / "resident_rl.jsonl",
                valid_for_ms=100,
            )
            try:
                step = runner.observe(
                    {"seq": 4, "stamp_ms": 9_990},
                    now_s=1.03,
                    action_stamp_ms=10_000,
                )
            finally:
                runner.close(action_stamp_ms=10_000)

        self.assertIsNotNone(step)
        motion_packet = self.packets()[-2]
        self.assertEqual(motion_packet["schema_version"], "stm32_velocity_command.v1")
        self.assertEqual(
            (
                motion_packet["boom_mps"],
                motion_packet["stick_mps"],
                motion_packet["bucket_mps"],
                motion_packet["swing_radps"],
            ),
            (0.025, -0.03, 0.04, -0.5),
        )

    def test_old_rl_generation_cannot_write_after_act_handoff_starts(self) -> None:
        self.activate()
        act_generation = self.sink.request_handoff(
            PolicyBinding("act_dig", ControlMode.MANUAL_ACTION),
            now_monotonic_ns=1_040_000_000,
        )
        writes_before = len(self.serial.writes)

        result = self.adapter.send(policy_action(sequence=8, stamp_ms=10_000))

        self.assertFalse(result.accepted)
        self.assertFalse(result.write_performed)
        self.assertEqual(result.reason, "handoff_in_progress")
        self.assertEqual(len(self.serial.writes), writes_before)
        self.assertEqual(self.sink.snapshot().generation, act_generation)

    def test_malformed_old_rl_packet_cannot_cancel_new_act_handoff(self) -> None:
        self.activate()
        act_binding = PolicyBinding("act_dig", ControlMode.MANUAL_ACTION)
        act_generation = self.sink.request_handoff(
            act_binding,
            now_monotonic_ns=1_040_000_000,
        )

        with self.assertRaisesRegex(ValueError, "strict finite JSON"):
            self.adapter.send(b"not-json")

        snapshot = self.sink.snapshot()
        self.assertEqual(snapshot.generation, act_generation)
        self.assertEqual(snapshot.target_binding, act_binding)

    def test_reordered_packet_stops_the_active_policy_with_zero(self) -> None:
        self.activate()
        self.adapter.send(policy_action(sequence=7, stamp_ms=9_980))

        with self.assertRaisesRegex(ValueError, "out-of-order"):
            self.adapter.send(policy_action(sequence=7, stamp_ms=10_000))

        packet = self.packets()[-1]
        self.assertEqual(packet["boom_mps"], 0.0)
        self.assertIsNone(self.adapter.generation)

    def test_expired_packet_stops_the_active_policy_with_zero(self) -> None:
        self.activate()

        with self.assertRaisesRegex(ValueError, "expired"):
            self.adapter.send(policy_action(sequence=8, stamp_ms=9_000))

        packet = self.packets()[-1]
        self.assertEqual(packet["boom_mps"], 0.0)
        self.assertIsNone(self.adapter.generation)

    def test_packet_contract_requires_canonical_action_order(self) -> None:
        self.activate()
        value = json.loads(policy_action(sequence=1, stamp_ms=10_000))
        value["action_order"] = ["swing", "boom", "stick", "bucket"]

        with self.assertRaisesRegex(ValueError, "action_order"):
            self.adapter.send(json.dumps(value).encode("utf-8"))
        self.assertEqual(self.packets()[-1]["boom_mps"], 0.0)


class ResidentPolicyCandidateAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = RecordingSerial()
        self.sink = ResidentCommandSink(self.serial, max_state_age_ms=200.0)
        self.sink.initialize(
            telemetry(receive_ns=990_000_000, command_seq=0, valid=False, mode=None)
        )
        self.adapter = ResidentPolicyCandidateAdapter(
            self.sink,
            binding=PolicyBinding("act_dig", ControlMode.MANUAL_ACTION),
            monotonic_ns=lambda: 1_030_000_000,
        )

    def packets(self) -> list[dict]:
        return [json.loads(payload.decode("ascii")) for payload in self.serial.writes]

    def activate(self) -> int:
        generation = self.adapter.begin_activation(now_monotonic_ns=1_000_000_000)
        claim = self.packets()[-1]
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_020_000_000,
                command_seq=claim["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        return generation

    @staticmethod
    def candidate(
        generation: int,
        *,
        source: str = "act_dig",
        mode: ControlMode = ControlMode.MANUAL_ACTION,
    ) -> bytes:
        return encode_motion_candidate(
            MotionCandidate(
                source=source,
                generation=generation,
                mode=mode,
                action=(0.1, -0.2, 0.3, -0.4),
                created_monotonic_ns=1_025_000_000,
                valid_until_monotonic_ns=1_080_000_000,
            )
        )

    def test_act_candidate_enters_manual_mode_without_reorder_or_scaling(self) -> None:
        generation = self.activate()

        result = self.adapter.send(self.candidate(generation))

        self.assertTrue(result.accepted)
        packet = self.packets()[-1]
        self.assertEqual(packet["schema_version"], "stm32_manual_command.v1")
        self.assertEqual(
            (packet["Y2"], packet["Y1"], packet["X2"], packet["X1"]),
            (0.1, -0.2, 0.3, -0.4),
        )

    def _reactivate(self) -> tuple[int, int]:
        old_generation = self.activate()
        self.sink.request_handoff(
            PolicyBinding("act_dig", ControlMode.MANUAL_ACTION),
            now_monotonic_ns=1_030_000_000,
        )
        current_generation = self.adapter.begin_activation(
            now_monotonic_ns=1_050_000_000
        )
        terminal_zero = self.packets()[-1]
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_060_000_000,
                command_seq=terminal_zero["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        target_zero = self.packets()[-1]
        self.sink.observe_telemetry(
            telemetry(
                receive_ns=1_070_000_000,
                command_seq=target_zero["command_seq"],
                valid=True,
                mode=ControlMode.MANUAL_ACTION,
            )
        )
        return old_generation, current_generation

    def test_old_generation_is_rejected_without_revoking_current_act_binding(self) -> None:
        old_generation, current_generation = self._reactivate()
        writes_before = len(self.serial.writes)

        result = self.adapter.send(self.candidate(old_generation))

        snapshot = self.sink.snapshot()
        self.assertFalse(result.accepted)
        self.assertFalse(result.write_performed)
        self.assertEqual(result.reason, "stale_generation")
        self.assertEqual(self.adapter.generation, current_generation)
        self.assertEqual(snapshot.generation, current_generation)
        self.assertEqual(
            snapshot.active_binding,
            PolicyBinding("act_dig", ControlMode.MANUAL_ACTION),
        )
        self.assertEqual(len(self.serial.writes), writes_before)

    def test_future_generation_is_rejected_without_revoking_current_act_binding(self) -> None:
        _, current_generation = self._reactivate()
        writes_before = len(self.serial.writes)

        result = self.adapter.send(self.candidate(current_generation + 1))

        snapshot = self.sink.snapshot()
        self.assertFalse(result.accepted)
        self.assertFalse(result.write_performed)
        self.assertEqual(result.reason, "stale_generation")
        self.assertEqual(self.adapter.generation, current_generation)
        self.assertEqual(snapshot.generation, current_generation)
        self.assertEqual(
            snapshot.active_binding,
            PolicyBinding("act_dig", ControlMode.MANUAL_ACTION),
        )
        self.assertEqual(len(self.serial.writes), writes_before)

    def test_current_generation_source_violation_revokes_current_act_binding(self) -> None:
        generation = self.activate()

        with self.assertRaisesRegex(ValueError, "source"):
            self.adapter.send(self.candidate(generation, source="stale_worker"))

        self.assertIsNone(self.adapter.generation)
        self.assertEqual(self.packets()[-1]["Y2"], 0.0)

    def test_malformed_old_act_packet_cannot_cancel_new_rl_handoff(self) -> None:
        self.activate()
        rl_binding = PolicyBinding("rl_follow", ControlMode.VELOCITY_REFERENCE)
        rl_generation = self.sink.request_handoff(
            rl_binding,
            now_monotonic_ns=1_040_000_000,
        )

        with self.assertRaisesRegex(ValueError, "strict finite JSON"):
            self.adapter.send(b"not-json")

        snapshot = self.sink.snapshot()
        self.assertEqual(snapshot.generation, rl_generation)
        self.assertEqual(snapshot.target_binding, rl_binding)


if __name__ == "__main__":
    unittest.main()
