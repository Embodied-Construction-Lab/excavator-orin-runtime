import tempfile
import unittest
from pathlib import Path

from edge_runtime.follow import EdgeFollowRuntime
from edge_runtime.kinematics import UrdfBucketTipKinematics
from tests.test_edge_kinematics import URDF


def machine_profile():
    return {
        "machine_id": "scale_excavator_v1",
        "action_order": ["boom", "stick", "bucket", "swing"],
        "actuators": {
            "boom": {
                "deploy_position_observation": {
                    "source": "stm32_absolute_cable_encoder",
                    "range": [0.07, 0.19],
                    "status": "firmware_safety_bounds",
                },
                "deploy_observation_sign": -1,
                "max_speed_positive": 0.04,
                "max_speed_negative": 0.02,
                "command_deadzone_positive_normalized": 0.0,
                "command_deadzone_negative_normalized": 0.0,
            },
            "stick": {
                "deploy_position_observation": {
                    "source": "stm32_absolute_cable_encoder",
                    "range": [0.06, 0.21],
                    "status": "firmware_safety_bounds",
                },
                "deploy_observation_sign": -1,
                "max_speed_positive": 0.05,
                "max_speed_negative": 0.05,
                "command_deadzone_positive_normalized": 0.0,
                "command_deadzone_negative_normalized": 0.0,
            },
            "bucket": {
                "deploy_position_observation": {
                    "source": "stm32_absolute_cable_encoder",
                    "range": [0.06, 0.16],
                    "status": "firmware_safety_bounds",
                },
                "deploy_observation_sign": -1,
                "max_speed_positive": 0.03,
                "max_speed_negative": 0.06,
                "command_deadzone_positive_normalized": 0.0,
                "command_deadzone_negative_normalized": 0.0,
            },
            "swing": {
                "deploy_observation_sign": -1,
                "max_speed_positive": 0.6,
                "max_speed_negative": 0.6,
                "command_deadzone_positive_normalized": 0.0,
                "command_deadzone_negative_normalized": 0.0,
            },
        },
        "observation_schema": {
            "total_dim": 38,
            "waypoint_lookahead": 3,
            "normalizers": {
                "position_normalizer": 1.13,
                "tip_velocity_scale": 0.05,
                "distance_normalizer": 1.13,
                "tube_radius": 0.04,
                "target_threshold": 0.25,
                "pitch_norm_deg": 180.0,
            },
        },
        "task_profile": {
            "bucket_pitch_targets_deg": {
                "MoveToDig": 70.0,
                "CarryMaterial": 180.0,
            }
        },
    }


def machine_state(sequence=10, stamp_ms=1000):
    return {
        "type": "machine_state_v1",
        "schema_version": "1.0",
        "seq": sequence,
        "stamp_ms": stamp_ms,
        "machine_id": "scale_excavator_v1",
        "safety": {
            "estop": False,
            "stm32_alive": True,
            "sensor_valid": True,
            "control_enabled": True,
            "fault_flags": [],
        },
        "actuator_state": {
            "boom": {"position_m": 0.1381, "velocity_mps": 0.001},
            "stick": {"position_m": 0.1562, "velocity_mps": -0.002},
            "bucket": {"position_m": 0.15848, "velocity_mps": 0.003},
            "swing": {"position_rad": 0.3473, "velocity_rad_s": -0.004},
        },
        "joint_state": {
            "position_rad": {
                "swing": 0.3473,
                "boom": 0.85172,
                "arm": 1.82963,
                "bucket": 2.00713,
            }
        },
    }


def trajectory():
    return {
        "schema_version": "trajectory_command.v1",
        "frame_id": "machine_root_ros",
        "task_mode": "CarryMaterial",
        "waypoints_base": [
            [1.2, -0.2, -0.2],
            [0.8, -0.1, 0.0],
            [0.7, 0.1, 0.2],
        ],
        "waypoint_count": 3,
        "target_threshold": 0.25,
        "tube_radius": 0.04,
    }


class RecordingPolicy:
    def __init__(self, action):
        self.action = list(action)
        self.observations = []

    def run(self, observation):
        self.observations.append(list(observation))
        return list(self.action)


class EdgeFollowRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        urdf_path = Path(self.temp_dir.name) / "waji.urdf"
        urdf_path.write_text(URDF, encoding="utf-8")
        self.kinematics = UrdfBucketTipKinematics.from_path(urdf_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_one_step_builds_38d_unity_observation_and_preserves_policy_signs(self):
        policy = RecordingPolicy([0.5, -0.5, 0.1, -0.2])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
        )

        step = runtime.step(machine_state(), now_s=1.0)

        self.assertEqual(len(step.observation), 38)
        self.assertEqual(policy.observations, [list(step.observation)])
        self.assertEqual(step.normalized_action, (0.5, -0.5, 0.1, -0.2))
        self.assertEqual(step.physical_action, (0.02, -0.025, 0.003, -0.12))
        self.assertEqual(step.waypoint_index, 0)
        self.assertFalse(step.completed)
        expected_ros_tip = (
            0.767383087910776,
            -0.299489293005342,
            -0.301587096786684,
        )
        for actual, expected in zip(step.bucket_tip_ros_m, expected_ros_tip):
            self.assertAlmostEqual(actual, expected, places=12)
        expected_unity_tip = (
            0.299489293005342,
            -0.301587096786684,
            0.767383087910776,
        )
        for actual, expected in zip(step.observation[9:12], expected_unity_tip):
            self.assertAlmostEqual(actual, expected / 1.13, places=12)
        self.assertEqual(step.observation[30:34], (0.0, 0.0, 0.0, 0.0))

    def test_next_step_uses_previous_policy_action_and_source_timestamp(self):
        policy = RecordingPolicy([0.25, -0.25, 0.5, -0.5])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
        )
        runtime.step(machine_state(stamp_ms=1000), now_s=1.0)

        step = runtime.step(machine_state(sequence=11, stamp_ms=1100), now_s=1.1)

        self.assertEqual(step.observation[30:34], (0.25, -0.25, 0.5, -0.5))
        self.assertEqual(step.source_stamp_ms, 1100)
        self.assertEqual(step.observation[12:15], (0.0, 0.0, 0.0))

    def test_invalid_machine_safety_is_rejected_before_policy(self):
        policy = RecordingPolicy([1.0, 1.0, 1.0, 1.0])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
        )
        state = machine_state()
        state["safety"] = dict(state["safety"], sensor_valid=False)

        with self.assertRaisesRegex(ValueError, "sensor_invalid"):
            runtime.step(state, now_s=1.0)

        self.assertEqual(policy.observations, [])


if __name__ == "__main__":
    unittest.main()
