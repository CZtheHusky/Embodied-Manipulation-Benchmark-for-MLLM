from __future__ import annotations

from abc import ABC, abstractmethod
import math
import random

from mllm_teleop_benchmark.config.schema import FrameTransformConfig


class ControlFrameTransform(ABC):
    visible_to_agent: bool

    @abstractmethod
    def transform_delta_position(self, delta_position: tuple[float, float, float]) -> tuple[float, float, float]:
        raise NotImplementedError

    @abstractmethod
    def transform_delta_rotation(self, delta_rotation_rpy: tuple[float, float, float]) -> tuple[float, float, float]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


class IdentityControlFrameTransform(ControlFrameTransform):
    def __init__(self, visible_to_agent: bool = True) -> None:
        self.visible_to_agent = visible_to_agent

    @classmethod
    def from_config(cls, config: FrameTransformConfig) -> "IdentityControlFrameTransform":
        return cls(config.visible_to_agent)

    def transform_delta_position(self, delta_position: tuple[float, float, float]) -> tuple[float, float, float]:
        return delta_position

    def transform_delta_rotation(self, delta_rotation_rpy: tuple[float, float, float]) -> tuple[float, float, float]:
        return delta_rotation_rpy

    def metadata(self) -> dict[str, object]:
        return {"type": "identity", "visible_to_agent": self.visible_to_agent}


class RotationControlFrameTransform(ControlFrameTransform):
    def __init__(
        self,
        rotation_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
        transform_type: str,
        visible_to_agent: bool = True,
        params: dict[str, object] | None = None,
    ) -> None:
        self.rotation_matrix = rotation_matrix
        self.transform_type = transform_type
        self.visible_to_agent = visible_to_agent
        self.params = dict(params or {})

    @classmethod
    def from_config(cls, config: FrameTransformConfig, seed: int | None = None) -> "RotationControlFrameTransform":
        rng = random.Random(seed if seed is not None else 0)
        if config.type == "rotation_2d":
            angle = _angle_from_params(config.params, "angle", 0.0)
            return cls(_rotation_z(angle), config.type, config.visible_to_agent, {"angle_radians": angle})
        if config.type == "random_rotation_2d":
            low, high = _range_from_params(config.params, "angle_range", (-math.pi, math.pi))
            angle = rng.uniform(low, high)
            return cls(_rotation_z(angle), config.type, config.visible_to_agent, {"sampling": "uniform_so2", "angle_radians": angle, "angle_range_radians": (low, high)})
        if config.type == "rotation_3d":
            roll, pitch, yaw = _rpy_from_params(config.params, (0.0, 0.0, 0.0))
            return cls(_rotation_rpy(roll, pitch, yaw), config.type, config.visible_to_agent, {"rpy_radians": (roll, pitch, yaw)})
        sampling = str(config.params.get("sampling", "uniform_so3"))
        if sampling == "uniform_so3":
            matrix, quat = _uniform_so3(rng)
            return cls(matrix, config.type, config.visible_to_agent, {"sampling": "uniform_so3", "quaternion_xyzw": quat})
        if sampling == "euler_ranges":
            roll_range = _range_from_params(config.params, "roll_range", (0.0, 0.0))
            pitch_range = _range_from_params(config.params, "pitch_range", (0.0, 0.0))
            yaw_range = _range_from_params(config.params, "yaw_range", (-math.pi, math.pi))
            roll = rng.uniform(*roll_range)
            pitch = rng.uniform(*pitch_range)
            yaw = rng.uniform(*yaw_range)
            return cls(
                _rotation_rpy(roll, pitch, yaw),
                config.type,
                config.visible_to_agent,
                {"sampling": "euler_ranges", "rpy_radians": (roll, pitch, yaw), "roll_range_radians": roll_range, "pitch_range_radians": pitch_range, "yaw_range_radians": yaw_range},
            )
        raise ValueError(f"Unsupported random_rotation_3d sampling mode: {sampling}")

    def transform_delta_position(self, delta_position: tuple[float, float, float]) -> tuple[float, float, float]:
        return _matvec(self.rotation_matrix, delta_position)

    def transform_delta_rotation(self, delta_rotation_rpy: tuple[float, float, float]) -> tuple[float, float, float]:
        return _matvec(self.rotation_matrix, delta_rotation_rpy)

    def metadata(self) -> dict[str, object]:
        if not self.visible_to_agent:
            return {"type": self.transform_type, "visible_to_agent": False}
        return {
            "type": self.transform_type,
            "visible_to_agent": True,
            "rotation_matrix": [list(row) for row in self.rotation_matrix],
            "params": self.params,
        }


def _angle_from_params(params: dict[str, object], prefix: str, default: float) -> float:
    if f"{prefix}_radians" in params:
        return float(params[f"{prefix}_radians"])
    if f"{prefix}_degrees" in params:
        return math.radians(float(params[f"{prefix}_degrees"]))
    return default


def _range_from_params(params: dict[str, object], name: str, default: tuple[float, float]) -> tuple[float, float]:
    degrees_key = f"{name}_degrees"
    radians_key = f"{name}_radians"
    if radians_key in params:
        values = (float(v) for v in params[radians_key])
    elif degrees_key in params:
        values = (math.radians(float(v)) for v in params[degrees_key])
    else:
        return default
    low, high = values
    return low, high


def _rpy_from_params(params: dict[str, object], default: tuple[float, float, float]) -> tuple[float, float, float]:
    if "rpy_radians" in params:
        values = (float(v) for v in params["rpy_radians"])
    elif "rpy_degrees" in params:
        values = (math.radians(float(v)) for v in params["rpy_degrees"])
    else:
        return default
    roll, pitch, yaw = values
    return roll, pitch, yaw


def _rotation_z(angle: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _uniform_so3(rng: random.Random) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], tuple[float, float, float, float]]:
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return _rotation_quaternion_xyzw(qx, qy, qz, qw), (qx, qy, qz, qw)


def _rotation_quaternion_xyzw(x: float, y: float, z: float, w: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _matvec(matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(sum(matrix[0][idx] * vector[idx] for idx in range(3))),
        float(sum(matrix[1][idx] * vector[idx] for idx in range(3))),
        float(sum(matrix[2][idx] * vector[idx] for idx in range(3))),
    )
