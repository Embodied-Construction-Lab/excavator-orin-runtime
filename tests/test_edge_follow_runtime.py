import math
import tempfile
import unittest
from pathlib import Path

from edge_runtime.follow import EdgeFollowRuntime
from edge_runtime.kinematics import UrdfBucketTipKinematics
from edge_runtime.trajectory import TrajectorySnapshot, WaypointTracker
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
        "target_threshold": 0.03,
        "tube_radius": 0.04,
    }


def mission():
    return {
        "schema_version": "excavation_mission.v1",
        "mission_id": "field_cycle_001",
        "frame_id": "machine_root_ros",
        "limits": {
            "waypoint_tolerance_m": 0.25,
            "waypoint_dwell_s": 0.0,
            "tracking_timeout_s": 60.0,
        },
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
            mission=mission(),
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

    def test_bucket_pitch_error_matches_unity_delta_angle_for_each_task_mode(self):
        observations = {}
        for task_mode in ("MoveToDig", "CarryMaterial"):
            configured_trajectory = dict(trajectory(), task_mode=task_mode)
            runtime = EdgeFollowRuntime(
                machine_profile=machine_profile(),
                kinematics=self.kinematics,
                policy=RecordingPolicy([0.0, 0.0, 0.0, 0.0]),
                trajectory=configured_trajectory,
                mission=mission(),
            )

            step = runtime.step(machine_state(), now_s=1.0)
            observations[task_mode] = step.observation

            current_pitch_deg = math.degrees(step.bucket_pitch_rad)
            target_pitch_deg = machine_profile()["task_profile"][
                "bucket_pitch_targets_deg"
            ][task_mode]
            unity_delta_deg = (
                (current_pitch_deg - target_pitch_deg + 180.0) % 360.0
            ) - 180.0
            self.assertAlmostEqual(
                step.observation[36],
                unity_delta_deg / 180.0,
                places=12,
            )

        self.assertEqual(observations["MoveToDig"][27:29], (1.0, 0.0))
        self.assertEqual(observations["CarryMaterial"][27:29], (0.0, 1.0))
        self.assertNotEqual(
            observations["MoveToDig"][36],
            observations["CarryMaterial"][36],
        )

    def test_next_step_uses_previous_policy_action_and_source_timestamp(self):
        policy = RecordingPolicy([0.25, -0.25, 0.5, -0.5])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
            mission=mission(),
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
            mission=mission(),
        )
        state = machine_state()
        state["safety"] = dict(state["safety"], sensor_valid=False)

        with self.assertRaisesRegex(ValueError, "sensor_invalid"):
            runtime.step(state, now_s=1.0)

        self.assertEqual(policy.observations, [])

    def test_waypoint_advance_uses_mission_tolerance_not_trajectory_threshold(self):
        snapshot = TrajectorySnapshot.from_mapping(trajectory())
        first = snapshot.waypoints[0]
        tracker = WaypointTracker(
            snapshot,
            waypoint_tolerance_m=mission()["limits"]["waypoint_tolerance_m"],
        )

        within_mission_tolerance = tracker.advance(
            (first[0] + 0.20, first[1], first[2])
        )
        outside_mission_tolerance = tracker.advance(
            (first[0] + 0.26, first[1], first[2])
        )

        self.assertEqual(snapshot.target_threshold_m, 0.03)
        self.assertEqual(within_mission_tolerance.current_index, 1)
        self.assertEqual(outside_mission_tolerance.current_index, 0)

    def test_episode_progress_uses_follow_monotonic_start_and_clamps_at_timeout(self):
        policy = RecordingPolicy([0.5, -0.5, 0.1, -0.2])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
            mission=mission(),
        )

        started = runtime.step(machine_state(sequence=20), now_s=100.0)
        middle = runtime.step(machine_state(sequence=21), now_s=130.0)
        ended = runtime.step(machine_state(sequence=22), now_s=160.0)
        beyond = runtime.step(machine_state(sequence=23), now_s=190.0)

        progress = [
            started.observation[29],
            middle.observation[29],
            ended.observation[29],
            beyond.observation[29],
        ]
        self.assertEqual(progress, [0.0, 0.5, 1.0, 1.0])
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(ended.result, "TIMEOUT")
        self.assertEqual(beyond.result, "TIMEOUT")
        self.assertEqual(ended.physical_action, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(beyond.physical_action, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(middle.follow_elapsed_s, 30.0)
        self.assertEqual(middle.tracking_timeout_s, 60.0)
        self.assertEqual(middle.waypoint_tolerance_m, 0.25)

    def test_new_follow_runtime_starts_episode_progress_at_zero(self):
        first_runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=RecordingPolicy([0.0, 0.0, 0.0, 0.0]),
            trajectory=trajectory(),
            mission=mission(),
        )
        first_runtime.step(machine_state(sequence=30), now_s=100.0)
        progressed = first_runtime.step(machine_state(sequence=31), now_s=130.0)

        new_runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=RecordingPolicy([0.0, 0.0, 0.0, 0.0]),
            trajectory=trajectory(),
            mission=mission(),
        )
        restarted = new_runtime.step(machine_state(sequence=1), now_s=500.0)

        self.assertEqual(progressed.observation[29], 0.5)
        self.assertEqual(restarted.observation[29], 0.0)

    def test_monotonic_time_regression_is_rejected(self):
        policy = RecordingPolicy([0.5, -0.5, 0.1, -0.2])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
            mission=mission(),
        )
        runtime.step(machine_state(sequence=40), now_s=100.0)

        with self.assertRaisesRegex(ValueError, "monotonic"):
            runtime.step(machine_state(sequence=41), now_s=99.0)

        self.assertEqual(len(policy.observations), 1)

    def test_completed_result_is_terminal_and_not_reclassified_as_timeout(self):
        deployed_trajectory = trajectory()
        deployed_trajectory["waypoints_base"] = [
            [
                0.767383087910776,
                -0.299489293005342,
                -0.301587096786684,
            ]
        ]
        deployed_trajectory["waypoint_count"] = 1
        policy = RecordingPolicy([0.5, -0.5, 0.1, -0.2])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=deployed_trajectory,
            mission=mission(),
        )

        completed = runtime.step(machine_state(sequence=50), now_s=100.0)
        still_completed = runtime.step(machine_state(sequence=51), now_s=170.0)

        self.assertEqual(completed.result, "COMPLETED")
        self.assertEqual(still_completed.result, "COMPLETED")
        self.assertEqual(still_completed.physical_action, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(policy.observations, [])

    def test_sequence_gap_does_not_interrupt_active_follow(self):
        policy = RecordingPolicy([0.5, -0.5, 0.1, -0.2])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
            mission=mission(),
        )

        runtime.step(machine_state(sequence=60), now_s=100.0)
        after_gap = runtime.step(machine_state(sequence=65), now_s=101.0)

        self.assertEqual(after_gap.result, "ACTIVE")
        self.assertEqual(after_gap.observation[29], 1.0 / 60.0)
        self.assertEqual(len(policy.observations), 2)

    def test_invalid_first_state_does_not_start_follow_clock(self):
        policy = RecordingPolicy([0.5, -0.5, 0.1, -0.2])
        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=self.kinematics,
            policy=policy,
            trajectory=trajectory(),
            mission=mission(),
        )
        invalid = machine_state(sequence=70)
        invalid["safety"] = dict(invalid["safety"], sensor_valid=False)

        with self.assertRaisesRegex(ValueError, "sensor_invalid"):
            runtime.step(invalid, now_s=50.0)
        started = runtime.step(machine_state(sequence=71), now_s=100.0)
        middle = runtime.step(machine_state(sequence=72), now_s=130.0)

        self.assertEqual(started.observation[29], 0.0)
        self.assertEqual(middle.observation[29], 0.5)

    def test_nonzero_waypoint_dwell_is_rejected_instead_of_silently_ignored(self):
        configured_mission = mission()
        configured_mission["limits"] = dict(
            configured_mission["limits"],
            waypoint_dwell_s=0.5,
        )

        with self.assertRaisesRegex(ValueError, "waypoint_dwell_s"):
            EdgeFollowRuntime(
                machine_profile=machine_profile(),
                kinematics=self.kinematics,
                policy=RecordingPolicy([0.0, 0.0, 0.0, 0.0]),
                trajectory=trajectory(),
                mission=configured_mission,
            )


if __name__ == "__main__":
    unittest.main()
