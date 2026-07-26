import unittest

from edge_runtime.trajectory import validate_trajectory_mission


def mission():
    return {
        "schema_version": "excavation_mission.v1",
        "mission_id": "field_cycle_001",
        "frame_id": "machine_root_ros",
        "targets": {
            "dig": {"position_m": [0.6, 0.3, 0.2]},
            "dump": {"position_m": [0.0, -0.8, -0.35]},
        },
        "limits": {
            "waypoint_tolerance_m": 0.25,
            "waypoint_dwell_s": 0.0,
            "tracking_timeout_s": 60.0,
        },
    }


def trajectory():
    sha256 = "a" * 64
    return {
        "schema_version": "trajectory_command.v1",
        "frame_id": "machine_root_ros",
        "task_mode": "CarryMaterial",
        "waypoints_base": [[0.0, -0.8, -0.35]],
        "waypoint_count": 1,
        "target_threshold": 0.03,
        "tube_radius": 0.04,
        "planning_scope": "execution_strict",
        "execution_eligible": True,
        "mission": {
            "id": "field_cycle_001",
            "sha256": sha256,
            "phase": "dump",
        },
    }


class TrajectoryMissionProvenanceTest(unittest.TestCase):
    def test_matching_mission_id_sha_phase_and_execution_flags_are_accepted(self):
        validate_trajectory_mission(
            trajectory(),
            mission(),
            mission_sha256="a" * 64,
        )

    def test_mismatched_or_non_executable_trajectory_is_rejected(self):
        cases = {
            "mission id": ("mission", "id", "wrong"),
            "mission sha256": ("mission", "sha256", "b" * 64),
            "mission phase": ("mission", "phase", "dig"),
            "planning_scope": (None, "planning_scope", "preview"),
            "execution_eligible": (None, "execution_eligible", False),
        }
        for expected, (section, key, value) in cases.items():
            with self.subTest(expected=expected):
                candidate = trajectory()
                if section is None:
                    candidate[key] = value
                else:
                    candidate[section] = dict(candidate[section], **{key: value})
                with self.assertRaisesRegex(ValueError, expected):
                    validate_trajectory_mission(
                        candidate,
                        mission(),
                        mission_sha256="a" * 64,
                    )


if __name__ == "__main__":
    unittest.main()
