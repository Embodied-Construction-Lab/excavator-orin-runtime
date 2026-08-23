import unittest
from pathlib import Path
from unittest import mock

from edge_runtime.trajectory_controller import (
    ACTION_ORDER,
    CartesianPTrajectoryController,
    OnnxRlTrajectoryControllerAdapter,
    TrajectoryControllerDescriptor,
    TrajectoryControllerFactory,
    build_trajectory_controller,
    build_trajectory_controller_builder,
)


class _OnnxPolicy:
    last_inference_ms = 4.25

    def __init__(self):
        self.observations = []

    def run(self, observation):
        self.observations.append(tuple(observation))
        return [0.1, -0.2, 0.3, -0.4]


class TrajectoryControllerSeamTest(unittest.TestCase):
    def test_deployment_builder_reuses_loaded_onnx_policy_but_not_controller(self):
        policy = _OnnxPolicy()
        with mock.patch(
            "edge_runtime.onnx_policy.OnnxPolicy",
            return_value=policy,
        ) as policy_type:
            builder = build_trajectory_controller_builder(
                "onnx_rl",
                onnx_path=Path("policy.onnx"),
            )
            first = builder()
            second = builder()

        self.assertIsNot(first, second)
        policy_type.assert_called_once_with(Path("policy.onnx"))
        first.compute_action([0.0] * 38)
        second.compute_action([1.0] * 38)
        self.assertEqual(len(policy.observations), 2)

    def test_onnx_rl_and_cartesian_p_share_the_four_axis_output_contract(self):
        onnx_policy = _OnnxPolicy()
        factory = TrajectoryControllerFactory(
            {
                "onnx_rl": lambda: OnnxRlTrajectoryControllerAdapter(onnx_policy),
                "cartesian_p": CartesianPTrajectoryController,
            }
        )
        observation = [0.0] * 38
        observation[15:18] = [0.25, -0.5, 0.75]
        observation[36] = 0.2

        rl = factory.create("onnx_rl")
        classical = factory.create("cartesian_p")
        rl_output = rl.compute_action(observation)
        classical_output = classical.compute_action(observation)

        self.assertEqual(rl.descriptor.action_order, ACTION_ORDER)
        self.assertEqual(classical.descriptor.action_order, ACTION_ORDER)
        self.assertEqual(rl_output.normalized_action, (0.1, -0.2, 0.3, -0.4))
        self.assertEqual(rl_output.inference_ms, 4.25)
        self.assertEqual(
            classical_output.normalized_action,
            (-0.5, 0.75, -0.2, 0.25),
        )
        self.assertEqual(onnx_policy.observations, [tuple(observation)])

    def test_factory_rejects_unknown_or_mislabelled_controller(self):
        class WrongController:
            descriptor = TrajectoryControllerDescriptor(
                backend_id="wrong", implementation="test.wrong"
            )

            def compute_action(self, _observation):
                return (0.0, 0.0, 0.0, 0.0)

            def reset(self):
                return None

        factory = TrajectoryControllerFactory({"onnx_rl": WrongController})

        with self.assertRaisesRegex(ValueError, "unknown trajectory controller"):
            factory.create("typo")
        with self.assertRaisesRegex(ValueError, "descriptor backend_id"):
            factory.create("onnx_rl")

    def test_deployment_factory_selects_classical_without_loading_an_onnx_file(self):
        controller = build_trajectory_controller(
            "cartesian_p", onnx_path=None
        )

        self.assertEqual(controller.descriptor.backend_id, "cartesian_p")
        self.assertEqual(
            controller.compute_action([0.0] * 38).normalized_action,
            (0.0, 0.0, -0.0, 0.0),
        )

    def test_deployment_factory_rejects_onnx_backend_without_model_path(self):
        with self.assertRaisesRegex(ValueError, "onnx_path is required"):
            build_trajectory_controller("onnx_rl", onnx_path=None)

    def test_checked_controller_rejects_a_backend_that_skips_the_result_contract(self):
        class TupleController:
            descriptor = TrajectoryControllerDescriptor(
                backend_id="tuple", implementation="test.tuple"
            )

            def compute_action(self, _observation):
                return (0.0, 0.0, 0.0, 0.0)

            def reset(self):
                return None

        controller = TrajectoryControllerFactory(
            {"tuple": TupleController}
        ).create("tuple")

        with self.assertRaisesRegex(ValueError, "TrajectoryControlOutput"):
            controller.compute_action([0.0] * 38)


if __name__ == "__main__":
    unittest.main()
