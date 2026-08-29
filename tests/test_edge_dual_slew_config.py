import json
import tempfile
import unittest
from pathlib import Path

from edge_runtime.remote import (
    EdgeFollowRuntimeFactory,
    FollowTrajectorySnapshot,
)
from edge_runtime.shadow import load_edge_runtime_config
from tests.test_edge_remote import trajectory_snapshot


def _remote_config() -> dict:
    return {
        "schema_version": "orin_edge_runtime.v1",
        "mode": "remote_control",
        "action_transport": "resident_sink",
        "machine_profile_path": "machine_profile.json",
        "urdf_path": "waji.urdf",
        "onnx_path": "policy.onnx",
        "trajectory_controller_backend": "onnx_rl",
        "mission_path": "excavation_cycle.json",
        "fixed_action_profile_path": "fixed_actions.json",
        "audit_path": "edge.jsonl",
        "action_valid_for_ms": 300,
        "follow_action_slew_rate_per_s": 3.0,
        "follow_action_startup_slew_rate_per_s": 4.0,
        "remote_behavior": {
            "bind_host": "0.0.0.0",
            "bind_port": 18083,
            "allowed_client_host": "192.168.50.1",
            "status_hz": 5.0,
            "status_timeout_s": 0.3,
        },
    }


class EdgeDualSlewConfigTest(unittest.TestCase):
    def test_runtime_factory_forwards_both_slew_rates(self):
        calls = []

        class RecordingRuntime:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        profile = {
            "machine_id": "scale_excavator_v1",
            "observation_schema": {
                "normalizers": {
                    "target_threshold": 0.03,
                    "tube_radius": 0.04,
                }
            },
        }
        mission = {
            "schema_version": "excavation_mission.v1",
            "mission_id": "mission-1",
            "frame_id": "machine_root_ros",
            "limits": {
                "waypoint_tolerance_m": 0.25,
                "waypoint_dwell_s": 0.0,
                "tracking_timeout_s": 60.0,
            },
        }
        factory = EdgeFollowRuntimeFactory(
            machine_profile=profile,
            kinematics=object(),
            policy=object(),
            mission=mission,
            mission_sha256="a" * 64,
            runtime_type=RecordingRuntime,
            action_slew_rate_per_s=3.0,
            action_startup_slew_rate_per_s=4.0,
            monotonic_clock=lambda: 123.5,
        )

        factory.create(
            FollowTrajectorySnapshot.from_mapping(
                trajectory_snapshot(),
                now_s=101.0,
            )
        )

        self.assertEqual(calls[0]["action_slew_rate_per_s"], 3.0)
        self.assertEqual(calls[0]["action_startup_slew_rate_per_s"], 4.0)

    def test_remote_config_loads_distinct_startup_and_steady_rates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edge.json"
            path.write_text(json.dumps(_remote_config()), encoding="utf-8")

            config = load_edge_runtime_config(path)

        self.assertEqual(config.follow_action_slew_rate_per_s, 3.0)
        self.assertEqual(config.follow_action_startup_slew_rate_per_s, 4.0)

    def test_startup_rate_requires_steady_rate(self):
        value = _remote_config()
        del value["follow_action_slew_rate_per_s"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edge.json"
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "follow_action_startup_slew_rate_per_s requires",
            ):
                load_edge_runtime_config(path)


if __name__ == "__main__":
    unittest.main()
