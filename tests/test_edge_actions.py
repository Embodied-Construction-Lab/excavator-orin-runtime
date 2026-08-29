import unittest

from edge_runtime.actions import (
    dual_rate_slew_limited_normalized_action,
    slew_limited_normalized_action,
)


class SlewLimitedNormalizedActionTest(unittest.TestCase):
    def test_startup_uses_faster_rate_then_reversal_uses_steady_rate(self):
        previous = (0.0, 0.0, 0.0, 0.0)
        startup_pending = (True, True, True, True)

        for expected in (0.2, 0.4, 0.6, 0.8, 1.0):
            action, startup_pending = dual_rate_slew_limited_normalized_action(
                (1.0, -1.0, 1.0, -1.0),
                previous,
                startup_pending=startup_pending,
                elapsed_s=0.05,
                startup_rate_per_s=4.0,
                steady_rate_per_s=3.0,
            )
            for actual, target in zip(
                action,
                (expected, -expected, expected, -expected),
            ):
                self.assertAlmostEqual(actual, target)
            previous = action

        self.assertEqual(startup_pending, (False, False, False, False))

        reversed_action, _ = dual_rate_slew_limited_normalized_action(
            (-1.0, 1.0, -1.0, 1.0),
            previous,
            startup_pending=startup_pending,
            elapsed_s=0.05,
            startup_rate_per_s=4.0,
            steady_rate_per_s=3.0,
        )
        for actual, expected in zip(
            reversed_action,
            (0.85, -0.85, 0.85, -0.85),
        ):
            self.assertAlmostEqual(actual, expected)

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
