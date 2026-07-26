"""Minimal ML-Agents ONNX Runtime wrapper for the Orin edge loop."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


class OnnxPolicyLoadError(RuntimeError):
    pass


class OnnxPolicy:
    OBSERVATION_INPUT = "obs_0"
    ACTION_OUTPUT = "deterministic_continuous_actions"

    def __init__(
        self,
        model_path: Path,
        *,
        session_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        model = Path(model_path)
        if not model.is_file():
            raise OnnxPolicyLoadError("ONNX model does not exist: %s" % model)
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise OnnxPolicyLoadError("numpy is required for edge inference") from exc
        self._numpy = np

        if session_factory is None:
            try:
                import onnxruntime as ort
            except ModuleNotFoundError as exc:
                raise OnnxPolicyLoadError(
                    "onnxruntime is required on Orin; install the JetPack-compatible wheel"
                ) from exc
            session_factory = ort.InferenceSession

        self._session = session_factory(
            str(model),
            providers=["CPUExecutionProvider"],
        )
        self._inputs = list(self._session.get_inputs())
        self._outputs = list(self._session.get_outputs())
        self._observation_input = self._find_tensor(
            self._inputs,
            self.OBSERVATION_INPUT,
            final_dimension=38,
        )
        self._action_output = self._find_tensor(
            self._outputs,
            self.ACTION_OUTPUT,
            final_dimension=4,
        )
        self._last_inference_ms = 0.0

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def run(self, observation: Sequence[float]) -> Sequence[float]:
        values = [float(value) for value in observation]
        if len(values) != 38 or not all(math.isfinite(value) for value in values):
            raise ValueError("ONNX observation must contain 38 finite values")
        array = self._numpy.asarray(values, dtype=self._numpy.float32).reshape(1, 38)
        feed = {}
        for item in self._inputs:
            if item.name == self._observation_input.name:
                feed[item.name] = array
            else:
                feed[item.name] = self._zero_input(item)
        started = time.perf_counter()
        try:
            output = self._session.run([self._action_output.name], feed)[0]
        finally:
            self._last_inference_ms = (time.perf_counter() - started) * 1000.0
        action = self._numpy.asarray(output, dtype=self._numpy.float32).reshape(-1)
        if action.size < 4:
            raise OnnxPolicyLoadError("ONNX action output has fewer than four values")
        values = [float(value) for value in action[:4]]
        if not all(math.isfinite(value) for value in values):
            raise OnnxPolicyLoadError("ONNX action output contains a non-finite value")
        return values

    def _zero_input(self, input_info: Any) -> Any:
        dtype = self._numpy.float32
        type_text = str(input_info.type)
        if "int64" in type_text:
            dtype = self._numpy.int64
        elif "int32" in type_text:
            dtype = self._numpy.int32
        elif "bool" in type_text:
            dtype = self._numpy.bool_
        shape = [
            int(value) if isinstance(value, int) and value > 0 else 1
            for value in (input_info.shape or [1])
        ]
        return self._numpy.zeros(shape, dtype=dtype)

    @staticmethod
    def _find_tensor(
        tensors: Sequence[Any],
        name: str,
        *,
        final_dimension: int,
    ) -> Any:
        for tensor in tensors:
            if tensor.name != name:
                continue
            shape = list(tensor.shape or [])
            if (
                str(tensor.type) != "tensor(float)"
                or len(shape) != 2
                or shape[-1] != final_dimension
            ):
                raise OnnxPolicyLoadError(
                    "ONNX tensor signature mismatch: %s type=%s shape=%s"
                    % (name, tensor.type, shape)
                )
            return tensor
        raise OnnxPolicyLoadError("ONNX tensor is missing: %s" % name)
