"""Dependency-free URDF forward kinematics for the deployed bucket tip chain."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Vector3, Vector3, Vector3]
Transform = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


@dataclass(frozen=True)
class BucketTipPose:
    frame_id: str
    child_frame_id: str
    position_m: Vector3
    orientation_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: Vector3
    rpy: Vector3
    axis: Vector3


class UrdfBucketTipKinematics:
    """Evaluate one URDF chain without ROS, TF or a second geometry model."""

    def __init__(
        self,
        *,
        root_link: str,
        tip_link: str,
        chain: Sequence[_Joint],
    ) -> None:
        if not chain:
            raise ValueError("URDF bucket-tip chain cannot be empty")
        self.root_link = root_link
        self.tip_link = tip_link
        self._chain = tuple(chain)
        self.revolute_joint_names = tuple(
            joint.name
            for joint in self._chain
            if joint.joint_type in ("revolute", "continuous")
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        root_link: str = "fk_root",
        tip_link: str = "bucket_tip",
    ) -> "UrdfBucketTipKinematics":
        try:
            document = ET.parse(str(Path(path)))
        except (OSError, ET.ParseError) as exc:
            raise ValueError("cannot read URDF %s: %s" % (path, exc)) from exc

        joints_by_child: Dict[str, _Joint] = {}
        for element in document.getroot().findall("joint"):
            joint = _parse_joint(element)
            if joint.child in joints_by_child:
                raise ValueError("URDF child link has multiple parents: %s" % joint.child)
            joints_by_child[joint.child] = joint

        reversed_chain = []
        current = tip_link
        visited = set()
        while current != root_link:
            if current in visited:
                raise ValueError("URDF chain contains a cycle at %s" % current)
            visited.add(current)
            joint = joints_by_child.get(current)
            if joint is None:
                raise ValueError(
                    "URDF has no chain from %s to %s; stopped at %s"
                    % (root_link, tip_link, current)
                )
            if joint.joint_type not in ("fixed", "revolute", "continuous"):
                raise ValueError(
                    "unsupported joint type in bucket-tip chain: %s=%s"
                    % (joint.name, joint.joint_type)
                )
            reversed_chain.append(joint)
            current = joint.parent

        return cls(
            root_link=root_link,
            tip_link=tip_link,
            chain=tuple(reversed(reversed_chain)),
        )

    def evaluate(self, joint_positions_rad: Mapping[str, float]) -> BucketTipPose:
        transform = _identity_transform()
        for joint in self._chain:
            transform = _multiply_transform(
                transform,
                _origin_transform(joint.xyz, joint.rpy),
            )
            if joint.joint_type in ("revolute", "continuous"):
                if joint.name not in joint_positions_rad:
                    raise ValueError("missing joint position: %s" % joint.name)
                angle = float(joint_positions_rad[joint.name])
                if not math.isfinite(angle):
                    raise ValueError("joint position must be finite: %s" % joint.name)
                transform = _multiply_transform(
                    transform,
                    _rotation_transform(_axis_angle(joint.axis, angle)),
                )

        rotation = tuple(
            tuple(transform[row][column] for column in range(3))
            for row in range(3)
        )
        return BucketTipPose(
            frame_id=self.root_link,
            child_frame_id=self.tip_link,
            position_m=(
                transform[0][3],
                transform[1][3],
                transform[2][3],
            ),
            orientation_xyzw=_quaternion_xyzw(rotation),
        )


def _parse_joint(element: ET.Element) -> _Joint:
    name = element.get("name", "").strip()
    joint_type = element.get("type", "").strip()
    parent_element = element.find("parent")
    child_element = element.find("child")
    if not name or parent_element is None or child_element is None:
        raise ValueError("URDF joint is missing name, parent or child")
    parent = parent_element.get("link", "").strip()
    child = child_element.get("link", "").strip()
    if not parent or not child:
        raise ValueError("URDF joint %s has an empty parent or child" % name)

    origin = element.find("origin")
    xyz = _parse_triplet(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _parse_triplet(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    axis_element = element.find("axis")
    axis = _parse_triplet(
        axis_element.get("xyz") if axis_element is not None else None,
        (1.0, 0.0, 0.0),
    )
    return _Joint(name, joint_type, parent, child, xyz, rpy, axis)


def _parse_triplet(text: Optional[str], default: Vector3) -> Vector3:
    if text is None:
        return default
    fields = text.split()
    if len(fields) != 3:
        raise ValueError("URDF vector must contain exactly three numbers: %r" % text)
    values = tuple(float(field) for field in fields)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("URDF vector contains a non-finite value: %r" % text)
    return values  # type: ignore[return-value]


def _identity_transform() -> Transform:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _origin_transform(xyz: Vector3, rpy: Vector3) -> Transform:
    roll, pitch, yaw = rpy
    rotation = _multiply_matrix3(
        _multiply_matrix3(_rotation_z(yaw), _rotation_y(pitch)),
        _rotation_x(roll),
    )
    return _transform(rotation, xyz)


def _rotation_transform(rotation: Matrix3) -> Transform:
    return _transform(rotation, (0.0, 0.0, 0.0))


def _transform(rotation: Matrix3, translation: Vector3) -> Transform:
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _multiply_transform(left: Transform, right: Transform) -> Transform:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(4))
            for column in range(4)
        )
        for row in range(4)
    )  # type: ignore[return-value]


def _multiply_matrix3(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _rotation_x(angle: float) -> Matrix3:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))


def _rotation_y(angle: float) -> Matrix3:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine))


def _rotation_z(angle: float) -> Matrix3:
    cosine, sine = math.cos(angle), math.sin(angle)
    return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))


def _axis_angle(axis: Vector3, angle: float) -> Matrix3:
    magnitude = math.sqrt(sum(value * value for value in axis))
    if magnitude <= 1e-12:
        raise ValueError("URDF joint axis cannot be zero")
    x, y, z = (value / magnitude for value in axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return (
        (
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ),
        (
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ),
        (
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ),
    )


def _quaternion_xyzw(rotation: Matrix3) -> Tuple[float, float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal = (rotation[0][0], rotation[1][1], rotation[2][2])
        index = max(range(3), key=lambda item: diagonal[item])
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0][1] + rotation[1][0]) / scale,
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[2][1] - rotation[1][2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
            quaternion = (
                (rotation[0][1] + rotation[1][0]) / scale,
                0.25 * scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                (rotation[0][2] - rotation[2][0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
            quaternion = (
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                0.25 * scale,
                (rotation[1][0] - rotation[0][1]) / scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]
