"""Primitive validation shared by the remote Follow contract."""

from __future__ import annotations

import math
import re
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be finite" % name)
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("%s must be finite" % name)
    return converted


def positive(name: str, value: Any) -> float:
    converted = finite(name, value)
    if converted <= 0.0:
        raise ValueError("%s must be positive" % name)
    return converted


def nonnegative(name: str, value: Any) -> float:
    converted = finite(name, value)
    if converted < 0.0:
        raise ValueError("%s must be nonnegative" % name)
    return converted


def text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s must be non-empty" % name)
    return value


def sha256(name: str, value: Any) -> str:
    converted = text(name, value)
    if not _SHA256.fullmatch(converted):
        raise ValueError("%s must be lowercase sha256" % name)
    return converted


def boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s must be boolean" % name)
    return value


def waypoints(value: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("waypoints must be a non-empty array")
    result = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError("waypoint[%d] must contain three values" % index)
        result.append(
            tuple(
                finite("waypoint[%d][%d]" % (index, axis), coordinate)
                for axis, coordinate in enumerate(point)
            )
        )
    return tuple(result)  # type: ignore[return-value]
