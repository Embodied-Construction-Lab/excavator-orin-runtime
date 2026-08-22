"""Adapters from existing policy-worker outputs to the resident motion seam."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import threading
import time
from typing import Callable, Mapping

from .resident_motion import (
    ACTION_ORDER,
    ControlMode,
    MotionCandidate,
    PolicyBinding,
    ZERO_ACTION,
)
from .resident_protocol import decode_motion_candidate
from .resident_sink import ResidentCommandSink, ResidentWriteResult


_POLICY_ACTION_FIELDS = {
    "type",
    "schema_version",
    "seq",
    "stamp_ms",
    "action_order",
    "action",
    "action_type",
    "valid_for_ms",
}
_COMPATIBILITY_ACTION_TYPE = "normalized_velocity_command"
_MAX_PACKET_BYTES = 8192
_MAX_FUTURE_SKEW_MS = 50


@dataclass(frozen=True)
class _AdapterState:
    generation: int | None = None
    last_sequence: int | None = None


class ResidentVelocityActionAdapter:
    """Socket-compatible sink for existing EdgeControlRunner packets.

    The legacy packet calls its physical values ``normalized_velocity_command``;
    this Adapter keeps that compatibility debt at the boundary and submits a
    typed physical-velocity candidate to the resident authority.
    """

    def __init__(
        self,
        command_sink: ResidentCommandSink,
        *,
        source: str,
        wall_time_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        binding = PolicyBinding(source, ControlMode.VELOCITY_REFERENCE)
        self._sink = command_sink
        self._binding = binding
        self._wall_time_ms = wall_time_ms
        self._monotonic_ns = monotonic_ns
        self._state = _AdapterState()
        self._lock = threading.Lock()

    @property
    def generation(self) -> int | None:
        with self._lock:
            return self._state.generation

    def begin_activation(self, *, now_monotonic_ns: int | None = None) -> int:
        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            current = self._state.generation
            if current is not None:
                snapshot = self._sink.snapshot()
                if snapshot.generation == current and self._binding in (
                    snapshot.active_binding,
                    snapshot.target_binding,
                ):
                    return current
            generation = self._sink.request_handoff(
                self._binding,
                now_monotonic_ns=now,
            )
            self._state = _AdapterState(generation=generation)
            return generation

    def send(self, payload: bytes) -> ResidentWriteResult:
        now_monotonic_ns = self._monotonic_ns()
        receive_wall_ms = self._wall_time_ms()
        with self._lock:
            generation = self._state.generation
            if generation is None:
                raise RuntimeError("resident RL policy is not activated")
            try:
                sequence, action, remaining_lease_ms = _decode_velocity_action(
                    payload,
                    receive_wall_ms=receive_wall_ms,
                    last_sequence=self._state.last_sequence,
                )
            except ValueError:
                try:
                    self._sink.request_source_stop(
                        self._binding,
                        generation=generation,
                        now_monotonic_ns=now_monotonic_ns,
                    )
                finally:
                    self._state = replace(self._state, generation=None)
                raise
            self._state = replace(self._state, last_sequence=sequence)
            candidate = MotionCandidate(
                source=self._binding.source,
                generation=generation,
                mode=self._binding.mode,
                action=action,
                created_monotonic_ns=now_monotonic_ns,
                valid_until_monotonic_ns=(
                    now_monotonic_ns + remaining_lease_ms * 1_000_000
                ),
            )
            return self._sink.submit_candidate(
                candidate,
                now_monotonic_ns=now_monotonic_ns,
            )

    def request_stop(self, *, now_monotonic_ns: int | None = None) -> bool:
        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            generation = self._state.generation
            if generation is None:
                return False
            stopped = self._sink.request_source_stop(
                self._binding,
                generation=generation,
                now_monotonic_ns=now,
            )
            self._state = replace(self._state, generation=None)
            return stopped


class ResidentPolicyCandidateAdapter:
    """Typed ingress for a resident policy worker such as ACT.

    The worker must echo the activation generation supplied by this Adapter.
    That token makes delayed output from the previous policy phase harmless at
    the final serial boundary.
    """

    def __init__(
        self,
        command_sink: ResidentCommandSink,
        *,
        binding: PolicyBinding,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._sink = command_sink
        self._binding = PolicyBinding(binding.source, binding.mode)
        self._monotonic_ns = monotonic_ns
        self._state = _AdapterState()
        self._lock = threading.Lock()

    @property
    def generation(self) -> int | None:
        with self._lock:
            return self._state.generation

    def begin_activation(self, *, now_monotonic_ns: int | None = None) -> int:
        now = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            current = self._state.generation
            if current is not None:
                snapshot = self._sink.snapshot()
                if snapshot.generation == current and self._binding in (
                    snapshot.active_binding,
                    snapshot.target_binding,
                ):
                    return current
            generation = self._sink.request_handoff(
                self._binding,
                now_monotonic_ns=now,
            )
            self._state = _AdapterState(generation=generation)
            return generation

    def send(self, payload: bytes) -> ResidentWriteResult:
        now_monotonic_ns = self._monotonic_ns()
        with self._lock:
            generation = self._state.generation
            if generation is None:
                raise RuntimeError("resident policy worker is not activated")
            try:
                candidate = decode_motion_candidate(payload)
            except ValueError:
                try:
                    self._sink.request_source_stop(
                        self._binding,
                        generation=generation,
                        now_monotonic_ns=now_monotonic_ns,
                    )
                finally:
                    self._state = replace(self._state, generation=None)
                raise

            # A fully decoded frame from another activation generation is a
            # delayed (or premature) worker output, not evidence that the
            # current generation violated its contract.  Reject it locally;
            # only the current generation is allowed to revoke current
            # authority.
            if candidate.generation != generation:
                return ResidentWriteResult(
                    accepted=False,
                    write_performed=False,
                    reason="stale_generation",
                    command_seq=None,
                    mode=self._binding.mode,
                    effective_action=ZERO_ACTION,
                )

            try:
                if candidate.source != self._binding.source:
                    raise ValueError("candidate source does not match activated policy")
                if candidate.mode is not self._binding.mode:
                    raise ValueError("candidate mode does not match activated policy")
            except ValueError:
                try:
                    self._sink.request_source_stop(
                        self._binding,
                        generation=generation,
                        now_monotonic_ns=now_monotonic_ns,
                    )
                finally:
                    self._state = replace(self._state, generation=None)
                raise
            return self._sink.submit_candidate(
                candidate,
                now_monotonic_ns=now_monotonic_ns,
            )


def _decode_velocity_action(
    payload: bytes,
    *,
    receive_wall_ms: int,
    last_sequence: int | None,
) -> tuple[int, tuple[float, float, float, float], int]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_PACKET_BYTES:
        raise ValueError("policy_action payload size is invalid")
    try:
        packet = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("policy_action is not strict finite JSON") from exc
    if not isinstance(packet, Mapping) or set(packet) != _POLICY_ACTION_FIELDS:
        raise ValueError("policy_action fields are invalid")
    if packet["type"] != "policy_action" or packet["schema_version"] != "1.0":
        raise ValueError("policy_action version is invalid")
    if packet["action_type"] != _COMPATIBILITY_ACTION_TYPE:
        raise ValueError("policy_action action_type is invalid")
    if packet["action_order"] != list(ACTION_ORDER):
        raise ValueError("policy_action action_order must be canonical")

    sequence = _integer("seq", packet["seq"], maximum=0xFFFFFFFFFFFFFFFF)
    if last_sequence is not None and sequence <= last_sequence:
        raise ValueError("policy_action is duplicate or out-of-order")
    stamp_ms = _integer("stamp_ms", packet["stamp_ms"])
    valid_for_ms = _integer("valid_for_ms", packet["valid_for_ms"])
    if stamp_ms - receive_wall_ms > _MAX_FUTURE_SKEW_MS:
        raise ValueError("policy_action is from the future")
    remaining_lease_ms = stamp_ms + valid_for_ms - receive_wall_ms
    if remaining_lease_ms < 0:
        raise ValueError("policy_action is expired")

    raw_action = packet["action"]
    if not isinstance(raw_action, list) or len(raw_action) != len(ACTION_ORDER):
        raise ValueError("policy_action action must contain four values")
    values = tuple(_finite_number(f"action[{index}]", value) for index, value in enumerate(raw_action))
    return sequence, values, remaining_lease_ms  # type: ignore[return-value]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _integer(name: str, value: object, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"policy_action {name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"policy_action {name} exceeds its allowed range")
    return value


def _finite_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"policy_action {name} must be finite")
    return float(value)
