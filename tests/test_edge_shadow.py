import json
import tempfile
import unittest
from pathlib import Path

import orin_state_sender

from edge_runtime.follow import EdgeFollowStep
from edge_runtime.shadow import EdgeShadowObserver, load_edge_runtime_config


class StubRuntime:
    def __init__(self):
        self.calls = []

    def step(self, machine_state, *, now_s):
        self.calls.append((machine_state, now_s))
        return EdgeFollowStep(
            source_seq=machine_state["seq"],
            source_stamp_ms=machine_state["stamp_ms"],
            waypoint_index=2,
            completed=False,
            bucket_tip_ros_m=(0.1, 0.2, 0.3),
            bucket_pitch_rad=0.4,
            observation=tuple(float(index) for index in range(38)),
            normalized_action=(0.1, -0.2, 0.3, -0.4),
            physical_action=(0.01, -0.02, 0.03, -0.04),
            commanded_normalized_action=(0.08, -0.16, 0.24, -0.32),
        )


class EdgeShadowObserverTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_observe_writes_audit_record_without_an_action_sink(self):
        audit_path = self.root / "logs" / "edge.jsonl"
        runtime = StubRuntime()
        observer = EdgeShadowObserver(runtime=runtime, audit_path=audit_path)
        self.addCleanup(observer.close)
        state = {"seq": 7, "stamp_ms": 1234}

        step = observer.observe(state, now_s=2.5)

        self.assertIsNotNone(step)
        self.assertEqual(runtime.calls, [(state, 2.5)])
        record = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(record["mode"], "shadow")
        self.assertEqual(record["source_seq"], 7)
        self.assertEqual(record["waypoint_index"], 2)
        self.assertEqual(record["normalized_action"], [0.1, -0.2, 0.3, -0.4])
        self.assertEqual(
            record["commanded_normalized_action"],
            [0.08, -0.16, 0.24, -0.32],
        )
        self.assertEqual(record["physical_action"], [0.01, -0.02, 0.03, -0.04])
        self.assertEqual(record["runtime_monotonic_s"], 2.5)
        self.assertIn("loop_elapsed_ms", record)
        self.assertNotIn("serial", record)
        self.assertNotIn("policy_action", record)

    def test_runtime_error_is_audited_and_does_not_escape_state_loop(self):
        class FailingRuntime:
            def step(self, machine_state, *, now_s):
                raise ValueError("sensor_invalid")

        audit_path = self.root / "edge.jsonl"
        observer = EdgeShadowObserver(
            runtime=FailingRuntime(),
            audit_path=audit_path,
        )
        self.addCleanup(observer.close)

        result = observer.observe({"seq": 8, "stamp_ms": 1300}, now_s=2.6)

        self.assertIsNone(result)
        record = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["reason"], "sensor_invalid")
        self.assertEqual(record["exception_type"], "ValueError")
        self.assertEqual(record["consecutive_rejections"], 1)

    def test_config_paths_are_resolved_relative_to_config_file(self):
        config_path = self.root / "deploy" / "edge_shadow.json"
        config_path.parent.mkdir()
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "orin_edge_runtime.v1",
                    "mode": "shadow",
                    "machine_profile_path": "machine_profile.json",
                    "urdf_path": "waji.urdf",
                    "onnx_path": "policy.onnx",
                    "trajectory_path": "trajectory.json",
                    "mission_path": "excavation_cycle.json",
                    "audit_path": "../logs/edge.jsonl",
                    "action_valid_for_ms": 300,
                }
            ),
            encoding="utf-8",
        )

        config = load_edge_runtime_config(config_path)

        self.assertEqual(
            config.machine_profile_path,
            config_path.parent / "machine_profile.json",
        )
        self.assertEqual(config.audit_path, self.root / "logs" / "edge.jsonl")
        self.assertIsNone(config.follow_action_slew_rate_per_s)
        self.assertEqual(config.action_transport, "loopback_udp")

    def test_control_mode_is_preserved_for_the_control_runner(self):
        config_path = self.root / "edge.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "orin_edge_runtime.v1",
                    "mode": "control",
                    "machine_profile_path": "machine_profile.json",
                    "urdf_path": "waji.urdf",
                    "onnx_path": "policy.onnx",
                    "trajectory_path": "trajectory.json",
                    "mission_path": "excavation_cycle.json",
                    "audit_path": "edge.jsonl",
                    "action_valid_for_ms": 300,
                }
            ),
            encoding="utf-8",
        )

        config = load_edge_runtime_config(config_path)

        self.assertEqual(config.mode, "control")

    def test_remote_control_config_has_no_static_trajectory_and_loads_server(self):
        config_path = self.root / "edge.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "orin_edge_runtime.v1",
                    "mode": "remote_control",
                    "machine_profile_path": "machine_profile.json",
                    "urdf_path": "waji.urdf",
                    "onnx_path": "policy.onnx",
                    "mission_path": "excavation_cycle.json",
                    "fixed_action_profile_path": "fixed_actions.json",
                    "audit_path": "edge.jsonl",
                    "action_valid_for_ms": 300,
                    "follow_action_slew_rate_per_s": 2.0,
                    "action_transport": "resident_sink",
                    "remote_behavior": {
                        "bind_host": "0.0.0.0",
                        "bind_port": 18083,
                        "allowed_client_host": "192.168.2.127",
                        "status_hz": 5.0,
                        "status_timeout_s": 0.3,
                    },
                }
            ),
            encoding="utf-8",
        )

        config = load_edge_runtime_config(config_path)

        self.assertEqual(config.mode, "remote_control")
        self.assertIsNone(config.trajectory_path)
        self.assertEqual(
            config.fixed_action_profile_path,
            self.root / "fixed_actions.json",
        )
        self.assertEqual(config.remote_behavior.bind_port, 18083)
        self.assertEqual(config.remote_behavior.status_hz, 5.0)
        self.assertEqual(config.follow_action_slew_rate_per_s, 2.0)
        self.assertEqual(config.action_transport, "resident_sink")

    def test_follow_action_slew_rate_must_be_positive_when_configured(self):
        config_path = self.root / "edge.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "orin_edge_runtime.v1",
                    "mode": "shadow",
                    "machine_profile_path": "machine_profile.json",
                    "urdf_path": "waji.urdf",
                    "onnx_path": "policy.onnx",
                    "trajectory_path": "trajectory.json",
                    "mission_path": "excavation_cycle.json",
                    "audit_path": "edge.jsonl",
                    "action_valid_for_ms": 300,
                    "follow_action_slew_rate_per_s": 0.0,
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "follow_action_slew_rate_per_s"):
            load_edge_runtime_config(config_path)

    def test_action_transport_must_be_known_when_configured(self):
        config_path = self.root / "edge.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "orin_edge_runtime.v1",
                    "mode": "control",
                    "machine_profile_path": "machine_profile.json",
                    "urdf_path": "waji.urdf",
                    "onnx_path": "policy.onnx",
                    "trajectory_path": "trajectory.json",
                    "mission_path": "excavation_cycle.json",
                    "audit_path": "edge.jsonl",
                    "action_valid_for_ms": 300,
                    "action_transport": "shared_memory_magic",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "action_transport"):
            load_edge_runtime_config(config_path)

    def test_config_loads_one_authoritative_mission_path(self):
        config_path = self.root / "edge.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "orin_edge_runtime.v1",
                    "mode": "shadow",
                    "machine_profile_path": "machine_profile.json",
                    "urdf_path": "waji.urdf",
                    "onnx_path": "policy.onnx",
                    "trajectory_path": "trajectory.json",
                    "mission_path": "excavation_cycle.json",
                    "audit_path": "edge.jsonl",
                    "action_valid_for_ms": 300,
                }
            ),
            encoding="utf-8",
        )

        config = load_edge_runtime_config(config_path)

        self.assertEqual(
            config.mission_path,
            self.root / "excavation_cycle.json",
        )

    def test_state_sender_accepts_one_optional_edge_shadow_config(self):
        args = orin_state_sender.parse_args(
            ["--edge-config", "deploy/edge_shadow.json"]
        )

        self.assertEqual(
            args.edge_config,
            Path("deploy/edge_shadow.json"),
        )

    def test_control_authorization_requires_exact_token(self):
        self.assertFalse(orin_state_sender.edge_control_authorized(""))
        self.assertFalse(orin_state_sender.edge_control_authorized("true"))
        self.assertTrue(
            orin_state_sender.edge_control_authorized(
                "ALLOW_EDGE_MACHINE_MOTION"
            )
        )
        self.assertTrue(orin_state_sender.edge_mode_controls_motion("control"))
        self.assertTrue(
            orin_state_sender.edge_mode_controls_motion("remote_control")
        )
        self.assertFalse(orin_state_sender.edge_mode_controls_motion("shadow"))


if __name__ == "__main__":
    unittest.main()
