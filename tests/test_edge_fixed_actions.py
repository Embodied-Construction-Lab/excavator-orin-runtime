import unittest

from edge_runtime.fixed_actions import (
    FixedActionProfile,
    FixedActionRuntime,
)


def machine_profile():
    actuator = {
        "range": [0.0, 1.0],
        "deploy_position_observation": {
            "source": "stm32_absolute_cable_encoder",
            "range": [0.0, 1.0],
            "status": "firmware_safety_bounds",
        },
        "deploy_observation_sign": -1,
        "max_speed_positive": 0.1,
        "max_speed_negative": 0.2,
        "command_deadzone_positive_normalized": 0.0,
        "command_deadzone_negative_normalized": 0.0,
    }
    return {
        "machine_id": "scale_excavator_v1",
        "action_order": ["boom", "stick", "bucket", "swing"],
        "actuators": {
            "boom": dict(actuator),
            "stick": dict(actuator),
            "bucket": dict(actuator),
            "swing": {
                **actuator,
                "max_speed_positive": 0.6,
                "max_speed_negative": 0.6,
            },
        },
    }


def profile():
    return FixedActionProfile.from_mapping(
        {
            "schema_version": "fixed_action_profile.v2",
            "profile_id": "test-profile",
            "machine_id": "scale_excavator_v1",
            "action_order": ["boom", "stick", "bucket", "swing"],
            "validation_status": "candidate",
            "validation_evidence": None,
            "machine_profile_sha256": "a" * 64,
            "urdf_sha256": "b" * 64,
            "controller": {
                "kp": 1.5,
                "min_action": 0.6,
                "max_action": 0.6,
                "tolerance": 0.03,
                "step_timeout_s": 6.0,
                "hold_s": 0.15,
            },
            "start_envelopes": {
                phase: {
                    "normalized_actuator_position": {
                        "boom": [-1.0, 1.0],
                        "stick": [-1.0, 1.0],
                        "bucket": [-1.0, 1.0],
                    },
                    "swing_rad": [-1.57, 1.57],
                }
                for phase in ("dig", "dump")
            },
            "actions": {
                "dig": [
                    {
                        "step_id": "dig",
                        "label": "dig",
                        "target_normalized_position": {"boom": 0.5},
                    }
                ],
                "dump": [
                    {
                        "step_id": "open",
                        "label": "open",
                        "target_normalized_position": {"bucket": 0.5},
                    },
                    {
                        "step_id": "recover",
                        "label": "recover",
                        "target_normalized_position": {"bucket": 0.0},
                    },
                ],
            },
        }
    )


def absolute_dig_profile():
    return FixedActionProfile.from_mapping(
        {
            "schema_version": "fixed_action_profile.v2",
            "profile_id": "test-absolute-profile",
            "machine_id": "scale_excavator_v1",
            "action_order": ["boom", "stick", "bucket", "swing"],
            "validation_status": "candidate",
            "validation_evidence": None,
            "machine_profile_sha256": "a" * 64,
            "urdf_sha256": "b" * 64,
            "controller": {
                "kp": 1.5,
                "min_action": 0.6,
                "max_action": 0.6,
                "tolerance": 0.03,
                "step_timeout_s": 6.0,
                "hold_s": 0.15,
            },
            "start_envelopes": {
                phase: {
                    "normalized_actuator_position": {
                        "boom": [-1.0, 1.0],
                        "stick": [-1.0, 1.0],
                        "bucket": [-1.0, 1.0],
                    },
                    "swing_rad": [-1.57, 1.57],
                }
                for phase in ("dig", "dump")
            },
            "actions": {
                "dig": [
                    {
                        "step_id": "prepare",
                        "label": "prepare",
                        "target_normalized_position": {
                            "boom": 0.35,
                            "stick": -0.15,
                            "bucket": 1.0,
                        },
                    },
                    {
                        "step_id": "curl",
                        "label": "curl",
                        "target_normalized_position": {
                            "stick": 0.15,
                            "bucket": -1.0,
                        },
                    },
                    {
                        "step_id": "lift",
                        "label": "lift",
                        "target_normalized_position": {
                            "boom": -0.35,
                        },
                    },
                ],
                "dump": [
                    {
                        "step_id": "open",
                        "label": "open",
                        "target_normalized_position": {"bucket": 1.0},
                    },
                    {
                        "step_id": "recover",
                        "label": "recover",
                        "target_normalized_position": {"bucket": -1.0},
                    },
                ],
            },
        }
    )


def state(*, boom_position=0.5, stick_position=0.5, bucket_position=0.5):
    return {
        "seq": 1,
        "actuator_state": {
            "boom": {"position_m": boom_position},
            "stick": {"position_m": stick_position},
            "bucket": {"position_m": bucket_position},
            "swing": {"position_rad": 0.0},
        },
    }


class FixedActionRuntimeTest(unittest.TestCase):
    def test_absolute_dig_targets_do_not_depend_on_start_pose(self):
        for bucket_position in (0.75, 0.25):
            with self.subTest(bucket_position=bucket_position):
                runtime = FixedActionRuntime(
                    profile=absolute_dig_profile(),
                    machine_profile=machine_profile(),
                    phase="dig",
                )

                preparing = runtime.step(
                    state(bucket_position=bucket_position),
                    now_s=0.0,
                )

                self.assertEqual(preparing.step_label, "prepare")
                self.assertEqual(
                    preparing.normalized_action,
                    (0.6, -0.6, 0.6, 0.0),
                )

    def test_absolute_dig_advances_through_prepare_curl_and_lift(self):
        runtime = FixedActionRuntime(
            profile=absolute_dig_profile(),
            machine_profile=machine_profile(),
            phase="dig",
        )
        prepared = state(
            boom_position=0.325,
            stick_position=0.575,
            bucket_position=0.0,
        )

        hold_prepare = runtime.step(prepared, now_s=0.0)
        self.assertEqual(hold_prepare.phase, "hold")

        curling = runtime.step(prepared, now_s=0.2)
        self.assertEqual(curling.step_label, "curl")
        self.assertEqual(curling.normalized_action, (0.0, 0.6, -0.6, 0.0))

        curled = state(
            boom_position=0.325,
            stick_position=0.425,
            bucket_position=1.0,
        )
        hold_curl = runtime.step(curled, now_s=0.3)
        self.assertEqual(hold_curl.phase, "hold")

        lifting = runtime.step(curled, now_s=0.5)
        self.assertEqual(lifting.step_label, "lift")
        self.assertEqual(lifting.normalized_action, (-0.6, 0.0, 0.0, 0.0))

    def test_dump_preserves_unity_command_sign_and_completes_with_zero(self):
        runtime = FixedActionRuntime(
            profile=profile(),
            machine_profile=machine_profile(),
            phase="dump",
        )

        opening = runtime.step(state(bucket_position=0.5), now_s=0.0)
        self.assertEqual(opening.result, "ACTIVE")
        self.assertEqual(opening.normalized_action, (0.0, 0.0, 0.6, 0.0))
        self.assertEqual(opening.physical_action, (0.0, 0.0, 0.06, 0.0))

        reached_open = runtime.step(state(bucket_position=0.25), now_s=0.1)
        self.assertEqual(reached_open.result, "ACTIVE")
        self.assertEqual(reached_open.phase, "hold")
        self.assertEqual(reached_open.physical_action, (0.0, 0.0, 0.0, 0.0))

        recovering = runtime.step(state(bucket_position=0.25), now_s=0.3)
        self.assertEqual(recovering.result, "ACTIVE")
        self.assertEqual(recovering.step_label, "recover")
        self.assertEqual(recovering.normalized_action, (0.0, 0.0, -0.6, 0.0))
        self.assertEqual(recovering.physical_action, (0.0, 0.0, -0.12, 0.0))

        reached_recover = runtime.step(state(bucket_position=0.5), now_s=0.4)
        self.assertEqual(reached_recover.phase, "hold")
        completed = runtime.step(state(bucket_position=0.5), now_s=0.6)
        self.assertEqual(completed.result, "COMPLETED")
        self.assertEqual(completed.physical_action, (0.0, 0.0, 0.0, 0.0))

    def test_step_timeout_is_terminal_and_zero(self):
        runtime = FixedActionRuntime(
            profile=profile(),
            machine_profile=machine_profile(),
            phase="dig",
        )
        runtime.step(state(), now_s=0.0)

        timed_out = runtime.step(state(), now_s=6.1)

        self.assertEqual(timed_out.result, "TIMEOUT")
        self.assertEqual(timed_out.reason_code, "STEP_TIMEOUT")
        self.assertEqual(timed_out.physical_action, (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
