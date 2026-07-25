import tempfile
import unittest
from pathlib import Path

from edge_runtime.kinematics import UrdfBucketTipKinematics


URDF = """\
<robot name="waji">
  <link name="fk_root"/>
  <joint name="fk_root_to_base" type="fixed">
    <origin xyz="-0.06 0 -0.18" rpy="0 0 0"/>
    <parent link="fk_root"/><child link="base_link"/>
  </joint>
  <link name="base_link"/>
  <joint name="swing_joint" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base_link"/><child link="swing_link"/>
    <axis xyz="0 0 -1"/>
  </joint>
  <link name="swing_link"/>
  <joint name="boom_joint" type="revolute">
    <origin xyz="0.05 0 0" rpy="0 0.52 0"/>
    <parent link="swing_link"/><child link="boom_link"/>
    <axis xyz="0 -1 0"/>
  </joint>
  <link name="boom_link"/>
  <joint name="arm_joint" type="revolute">
    <origin xyz="0.78 0 0" rpy="0 2.95 0"/>
    <parent link="boom_link"/><child link="arm_link"/>
    <axis xyz="0 -1 0"/>
  </joint>
  <link name="arm_link"/>
  <joint name="bucket_joint" type="revolute">
    <origin xyz="0.35 0 0" rpy="0 3.67 0"/>
    <parent link="arm_link"/><child link="bucket_link"/>
    <axis xyz="0 -1 0"/>
  </joint>
  <link name="bucket_link"/>
  <joint name="bucket_to_tip" type="fixed">
    <origin xyz="0.2 0 0" rpy="0 1.1 0"/>
    <parent link="bucket_link"/><child link="bucket_tip"/>
  </joint>
  <link name="bucket_tip"/>
</robot>
"""


class UrdfBucketTipKinematicsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.urdf_path = Path(self.temp_dir.name) / "waji.urdf"
        self.urdf_path.write_text(URDF, encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_matches_ros_kdl_golden_pose_at_measured_joint_state(self):
        kinematics = UrdfBucketTipKinematics.from_path(self.urdf_path)

        pose = kinematics.evaluate(
            {
                "swing_joint": 0.3473,
                "boom_joint": 0.85172,
                "arm_joint": 1.82963,
                "bucket_joint": 2.00713,
            }
        )

        self.assertEqual(pose.frame_id, "fk_root")
        self.assertEqual(pose.child_frame_id, "bucket_tip")
        expected_position = (
            0.767383087910776,
            -0.299489293005342,
            -0.301587096786684,
        )
        expected_quaternion = (
            0.169162062341057,
            0.964343863495717,
            0.035165903499455,
            -0.200470618380204,
        )
        for actual, expected in zip(pose.position_m, expected_position):
            self.assertAlmostEqual(actual, expected, places=12)
        for actual, expected in zip(pose.orientation_xyzw, expected_quaternion):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_joint_values_are_applied_by_name_not_input_order(self):
        kinematics = UrdfBucketTipKinematics.from_path(self.urdf_path)

        pose = kinematics.evaluate(
            {
                "bucket_joint": 0.7,
                "arm_joint": -0.3,
                "swing_joint": -0.4,
                "boom_joint": 0.2,
            }
        )

        expected = (0.552942542252939, 0.259147950340672, -0.33076420570464)
        for actual, value in zip(pose.position_m, expected):
            self.assertAlmostEqual(actual, value, places=12)

    def test_missing_revolute_joint_is_rejected(self):
        kinematics = UrdfBucketTipKinematics.from_path(self.urdf_path)

        with self.assertRaisesRegex(ValueError, "bucket_joint"):
            kinematics.evaluate(
                {
                    "swing_joint": 0.0,
                    "boom_joint": 0.0,
                    "arm_joint": 0.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
