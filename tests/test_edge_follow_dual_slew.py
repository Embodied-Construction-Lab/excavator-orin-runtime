import tempfile
import unittest
from pathlib import Path

from edge_runtime.follow import EdgeFollowRuntime
from edge_runtime.kinematics import UrdfBucketTipKinematics
from tests.test_edge_follow_runtime import (
    RecordingPolicy,
    machine_profile,
    machine_state,
    mission,
    trajectory,
)
from tests.test_edge_kinematics import URDF


class EdgeFollowDualSlewTest(unittest.TestCase):
    def test_runtime_uses_startup_rate_until_first_target_is_reached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "waji.urdf"
            urdf_path.write_text(URDF, encoding="utf-8")
            runtime = EdgeFollowRuntime(
                machine_profile=machine_profile(),
                kinematics=UrdfBucketTipKinematics.from_path(urdf_path),
                policy=RecordingPolicy([1.0, -1.0, 1.0, -1.0]),
                trajectory=trajectory(),
                mission=mission(),
                action_slew_rate_per_s=3.0,
                action_startup_slew_rate_per_s=4.0,
                slew_started_monotonic_s=10.0,
            )

            first = runtime.step(machine_state(sequence=10), now_s=10.05)
            second = runtime.step(machine_state(sequence=11), now_s=10.10)

        for actual, expected in zip(
            first.commanded_normalized_action,
            (0.2, -0.2, 0.2, -0.2),
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            second.commanded_normalized_action,
            (0.4, -0.4, 0.4, -0.4),
        ):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
