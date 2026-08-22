"""Single-sequence STM32 command encoding for the Resident Mission Runtime.

The unified firmware accepts two intentionally different wire contracts:
physical velocity references for RL and normalized manual actions for ACT.  A
single encoder owns their shared uint32 command sequence so a semantic handoff
cannot accidentally restart or fork the STM32 sequence history.
"""

from __future__ import annotations

import json
import math

from .resident_motion import ACTION_ORDER, ControlMode


MANUAL_COMMAND_SCHEMA_VERSION = "stm32_manual_command.v1"
VELOCITY_COMMAND_SCHEMA_VERSION = "stm32_velocity_command.v1"
UINT32_MAX = 0xFFFFFFFF


class Stm32ResidentCommandEncoder:
    """Encode both firmware modes while retaining one shared sequence."""

    def __init__(self) -> None:
        self._next_sequence = 0

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def synchronize(self, *, command_rx_seq: int, command_received: bool) -> int:
        if (
            isinstance(command_rx_seq, bool)
            or not isinstance(command_rx_seq, int)
            or not 0 <= command_rx_seq <= UINT32_MAX
        ):
            raise ValueError("command_rx_seq must be a uint32")
        if not isinstance(command_received, bool):
            raise ValueError("command_received must be boolean")
        self._next_sequence = (
            (command_rx_seq + 1) & UINT32_MAX if command_received else 0
        )
        return self._next_sequence

    def encode(
        self,
        *,
        mode: ControlMode,
        action: tuple[float, float, float, float],
        monotonic_ns: int,
    ) -> bytes:
        command_mode = ControlMode(mode)
        values = _canonical_action(action)
        timestamp = _monotonic_ns(monotonic_ns)
        sequence = self._next_sequence

        if command_mode is ControlMode.MANUAL_ACTION:
            if not all(-1.000001 <= value <= 1.000001 for value in values):
                raise ValueError("manual action must remain within [-1, 1]")
            boom, stick, bucket, swing = values
            payload = {
                "schema_version": MANUAL_COMMAND_SCHEMA_VERSION,
                "X1": swing,
                "Y1": stick,
                "Z1": 0.0,
                "X2": bucket,
                "Y2": boom,
                "Z2": 0.0,
                "command_seq": sequence,
                "command_source_stamp_ms": (timestamp // 1_000_000) & UINT32_MAX,
            }
        else:
            boom, stick, bucket, swing = values
            payload = {
                "schema_version": VELOCITY_COMMAND_SCHEMA_VERSION,
                "boom_mps": boom,
                "stick_mps": stick,
                "bucket_mps": bucket,
                "swing_radps": swing,
                "command_seq": sequence,
                "command_source_stamp_ms": (timestamp // 1_000_000) & UINT32_MAX,
            }

        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        self._next_sequence = (sequence + 1) & UINT32_MAX
        return encoded


def _canonical_action(
    action: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not isinstance(action, tuple) or len(action) != len(ACTION_ORDER):
        raise ValueError("STM32 command action must be the canonical four-axis tuple")
    values = []
    for value in action:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("STM32 command action must contain finite numeric values")
        values.append(float(value))
    values = tuple(values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("STM32 command action must contain finite values")
    return values  # type: ignore[return-value]


def _monotonic_ns(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("STM32 command timestamp must be nonnegative integer nanoseconds")
    return value
