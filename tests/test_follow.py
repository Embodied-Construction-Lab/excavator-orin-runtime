import unittest
from types import SimpleNamespace

from edge_runtime.follow import EdgeFollowRuntime
from tests.test_edge_follow_runtime import (
    RecordingPolicy,
    machine_profile,
    machine_state,
    mission,
    trajectory,
)


class EdgeFollowTrackingEvidenceTest(unittest.TestCase):
    def test_step_exposes_the_true_current_waypoint_after_tracker_advance(self):
        configured_trajectory = trajectory()
        first, second, _final = configured_trajectory["waypoints_base"]

        class FixedPoseKinematics:
            root_link = "fk_root"

            def evaluate(self, _joints):
                return SimpleNamespace(
                    position_m=tuple(first),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                )

        runtime = EdgeFollowRuntime(
            machine_profile=machine_profile(),
            kinematics=FixedPoseKinematics(),
            policy=RecordingPolicy([0.0, 0.0, 0.0, 0.0]),
            trajectory=configured_trajectory,
            mission=mission(),
        )

        step = runtime.step(machine_state(sequence=1), now_s=1.0)

        self.assertEqual(step.waypoint_index, 1)
        self.assertEqual(step.reference_waypoint_ros_m, tuple(second))


if __name__ == "__main__":
    unittest.main()
