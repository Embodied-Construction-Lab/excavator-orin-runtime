import unittest

from edge_runtime.actions import slew_limited_normalized_action


class SlewLimitedNormalizedActionTest(unittest.TestCase):
    def test_limits_each_axis_delta_without_changing_target_sign_or_steady_state(self):
        previous = (0.2, -0.2, 0.0, 0.8)
        target = (1.0, -1.0, -0.1, 0.5)

        limited = slew_limited_normalized_action(
            target,
            previous,
            elapsed_s=0.1,
            max_rate_per_s=2.0,
        )

        for actual, expected in zip(limited, (0.4, -0.4, -0.1, 0.6)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            slew_limited_normalized_action(
                target,
                limited,
                elapsed_s=1.0,
                max_rate_per_s=2.0,
            ),
            target,
        )

    def test_reversal_crosses_zero_instead_of_jumping_to_opposite_direction(self):
        limited = slew_limited_normalized_action(
            (-1.0, 1.0, 0.0, 0.0),
            (1.0, -1.0, 0.0, 0.0),
            elapsed_s=0.1,
            max_rate_per_s=2.0,
        )

        self.assertEqual(limited, (0.8, -0.8, 0.0, 0.0))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "four"):
            slew_limited_normalized_action(
                (1.0,),
                (0.0, 0.0, 0.0, 0.0),
                elapsed_s=0.1,
                max_rate_per_s=2.0,
            )
        with self.assertRaisesRegex(ValueError, "elapsed_s"):
            slew_limited_normalized_action(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                elapsed_s=-0.1,
                max_rate_per_s=2.0,
            )
        with self.assertRaisesRegex(ValueError, "max_rate_per_s"):
            slew_limited_normalized_action(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0),
                elapsed_s=0.1,
                max_rate_per_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
