import tempfile
import unittest
from pathlib import Path

import numpy as np

from edge_runtime.onnx_policy import OnnxPolicy


class TensorInfo:
    def __init__(self, name, shape, type_name="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_name


class FakeSession:
    def __init__(self):
        self.feeds = []

    def get_inputs(self):
        return [
            TensorInfo("obs_0", [1, 38]),
            TensorInfo("recurrent_in", [1, 2]),
        ]

    def get_outputs(self):
        return [TensorInfo("deterministic_continuous_actions", [1, 4])]

    def run(self, output_names, feed):
        self.feeds.append((output_names, feed))
        return [np.asarray([[0.5, -0.25, 1.0, -1.0]], dtype=np.float32)]


class OnnxPolicyTest(unittest.TestCase):
    def test_returns_deterministic_four_axis_action_without_sign_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "policy.onnx"
            model.write_bytes(b"test fixture")
            session = FakeSession()
            policy = OnnxPolicy(
                model,
                session_factory=lambda path, providers: session,
            )

            action = policy.run([0.0] * 38)

        self.assertEqual(action, [0.5, -0.25, 1.0, -1.0])
        output_names, feed = session.feeds[0]
        self.assertEqual(
            output_names,
            ["deterministic_continuous_actions"],
        )
        self.assertEqual(feed["obs_0"].shape, (1, 38))
        self.assertTrue(np.all(feed["recurrent_in"] == 0.0))


if __name__ == "__main__":
    unittest.main()
