"""Replaceable trajectory-controller seam for Orin Follow behavior.

Controllers consume the frozen 38D trajectory observation and return one
normalized velocity reference in the authoritative axis order.  Conversion to
physical velocity and all motion authority remain in the existing Follow and
resident control Modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, Tuple


ACTION_ORDER = ("boom", "stick", "bucket", "swing")
OUTPUT_SEMANTICS = "normalized_velocity_reference"


@dataclass(frozen=True)
class TrajectoryControllerDescriptor:
    backend_id: str
    implementation: str
    action_order: Tuple[str, str, str, str] = ACTION_ORDER
    output_semantics: str = OUTPUT_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError("trajectory controller backend_id must be non-empty text")
        if not isinstance(self.implementation, str) or not self.implementation.strip():
            raise ValueError("trajectory controller implementation must be non-empty text")
        if tuple(self.action_order) != ACTION_ORDER:
            raise ValueError("trajectory controller action_order is not authoritative")
        if self.output_semantics != OUTPUT_SEMANTICS:
            raise ValueError("trajectory controller output semantics are invalid")


@dataclass(frozen=True)
class TrajectoryControlOutput:
    normalized_action: Tuple[float, float, float, float]
    inference_ms: float


class TrajectoryController(Protocol):
    @property
    def descriptor(self) -> TrajectoryControllerDescriptor:
        ...

    def compute_action(
        self, observation: Sequence[float]
    ) -> TrajectoryControlOutput:
        ...

    def reset(self) -> None:
        ...


class OnnxRlTrajectoryControllerAdapter:
    """Adapter from the deployed ML-Agents ONNX wrapper to the controller seam."""

    descriptor = TrajectoryControllerDescriptor(
        backend_id="onnx_rl",
        implementation="edge_runtime.OnnxPolicy",
    )

    def __init__(self, policy: Any) -> None:
        if policy is None or not callable(getattr(policy, "run", None)):
            raise ValueError("ONNX RL policy must provide run(observation)")
        self._policy = policy

    def compute_action(
        self, observation: Sequence[float]
    ) -> TrajectoryControlOutput:
        action = self._policy.run(_observation(observation))
        inference_ms = float(getattr(self._policy, "last_inference_ms", 0.0))
        return TrajectoryControlOutput(
            normalized_action=_normalized_action(action),
            inference_ms=_inference_ms(inference_ms),
        )

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()


class CartesianPTrajectoryController:
    """Deterministic Cartesian P reference controller for ablation experiments.

    The current 38D observation contains the first look-ahead waypoint error at
    indices 15:18 in Unity semantic axes ``(lateral, vertical, forward)`` and
    the normalized bucket-pitch error at index 36.  This reference maps those
    errors to ``(boom, stick, bucket, swing)`` without hidden sign changes.
    It is intentionally simple and must be commissioned like any new policy
    before use on powered hydraulics.
    """

    descriptor = TrajectoryControllerDescriptor(
        backend_id="cartesian_p",
        implementation="edge_runtime.CartesianPTrajectoryController",
    )

    def __init__(
        self,
        *,
        boom_gain: float = 1.0,
        stick_gain: float = 1.0,
        bucket_gain: float = 1.0,
        swing_gain: float = 1.0,
        max_normalized_action: float = 1.0,
    ) -> None:
        gains = (boom_gain, stick_gain, bucket_gain, swing_gain)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in gains
        ):
            raise ValueError("Cartesian P gains must be finite and nonnegative")
        if (
            isinstance(max_normalized_action, bool)
            or not isinstance(max_normalized_action, (int, float))
            or not math.isfinite(float(max_normalized_action))
            or not 0.0 < float(max_normalized_action) <= 1.0
        ):
            raise ValueError("max_normalized_action must be in (0, 1]")
        self._gains = tuple(float(value) for value in gains)
        self._limit = float(max_normalized_action)

    def compute_action(
        self, observation: Sequence[float]
    ) -> TrajectoryControlOutput:
        values = _observation(observation)
        lateral, vertical, forward = values[15:18]
        pitch_error = values[36]
        targets = (
            self._gains[0] * vertical,
            self._gains[1] * forward,
            -self._gains[2] * pitch_error,
            self._gains[3] * lateral,
        )
        action = tuple(
            max(-self._limit, min(self._limit, value)) for value in targets
        )
        return TrajectoryControlOutput(
            normalized_action=_normalized_action(action),
            inference_ms=0.0,
        )

    def reset(self) -> None:
        return None


class _CheckedTrajectoryController:
    def __init__(self, adapter: TrajectoryController) -> None:
        self._adapter = adapter

    @property
    def descriptor(self) -> TrajectoryControllerDescriptor:
        return self._adapter.descriptor

    def compute_action(
        self, observation: Sequence[float]
    ) -> TrajectoryControlOutput:
        output = self._adapter.compute_action(_observation(observation))
        if not isinstance(output, TrajectoryControlOutput):
            raise ValueError(
                "trajectory controller must return TrajectoryControlOutput"
            )
        return TrajectoryControlOutput(
            normalized_action=_normalized_action(output.normalized_action),
            inference_ms=_inference_ms(output.inference_ms),
        )

    def reset(self) -> None:
        self._adapter.reset()


class TrajectoryControllerFactory:
    """Strict backend selection used by deployment configuration."""

    def __init__(
        self,
        builders: Mapping[str, Callable[[], TrajectoryController]],
    ) -> None:
        copied = dict(builders)
        if not copied or any(
            not isinstance(name, str) or not name.strip() or not callable(builder)
            for name, builder in copied.items()
        ):
            raise ValueError(
                "trajectory controller builders must be a non-empty named mapping"
            )
        self._builders = MappingProxyType(copied)

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def create(self, backend_id: str) -> TrajectoryController:
        if backend_id not in self._builders:
            raise ValueError(
                "unknown trajectory controller %r; expected one of %s"
                % (backend_id, list(self.backend_ids))
            )
        adapter = self._builders[backend_id]()
        descriptor = getattr(adapter, "descriptor", None)
        if not isinstance(descriptor, TrajectoryControllerDescriptor):
            raise ValueError("trajectory controller descriptor is missing or invalid")
        if descriptor.backend_id != backend_id:
            raise ValueError(
                "trajectory controller descriptor backend_id does not match selection"
            )
        return _CheckedTrajectoryController(adapter)


def build_trajectory_controller(
    backend_id: str,
    *,
    onnx_path: Path | None,
) -> TrajectoryController:
    """Build a configured controller while keeping selection in one factory."""

    return build_trajectory_controller_builder(
        backend_id,
        onnx_path=onnx_path,
    )()


def build_trajectory_controller_builder(
    backend_id: str,
    *,
    onnx_path: Path | None,
) -> Callable[[], TrajectoryController]:
    """Load deployment inference assets once and build fresh controller adapters.

    The ONNX session is the expensive inference asset and is shared across the
    executor's serialized Follow behaviors. Each call still returns a new
    checked adapter so controller-local state cannot leak between behaviors.
    """

    if backend_id not in {"onnx_rl", "cartesian_p"}:
        raise ValueError(
            "unknown trajectory controller %r; expected one of %s"
            % (backend_id, ["cartesian_p", "onnx_rl"])
        )

    if backend_id == "onnx_rl":
        if onnx_path is None:
            raise ValueError("onnx_path is required for onnx_rl")
        from .onnx_policy import OnnxPolicy

        policy = OnnxPolicy(Path(onnx_path))

        def build_onnx_rl() -> TrajectoryController:
            return OnnxRlTrajectoryControllerAdapter(policy)

        builders = {"onnx_rl": build_onnx_rl}
    else:
        builders = {"cartesian_p": CartesianPTrajectoryController}

    factory = TrajectoryControllerFactory(builders)
    return lambda: factory.create(backend_id)


def _observation(values: Sequence[float]) -> Tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("trajectory observation must contain 38 finite values") from exc
    if len(raw) != 38 or any(isinstance(value, bool) for value in raw):
        raise ValueError("trajectory observation must contain 38 finite values")
    converted = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("trajectory observation must contain 38 finite values")
    return converted


def _normalized_action(values: Any) -> Tuple[float, float, float, float]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("trajectory controller must return four normalized values") from exc
    if len(raw) != 4 or any(isinstance(value, bool) for value in raw):
        raise ValueError("trajectory controller must return four normalized values")
    converted = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) and -1.000001 <= value <= 1.000001 for value in converted):
        raise ValueError("trajectory controller must return four normalized values")
    return converted  # type: ignore[return-value]


def _inference_ms(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("trajectory controller inference_ms is invalid")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError("trajectory controller inference_ms is invalid")
    return converted
