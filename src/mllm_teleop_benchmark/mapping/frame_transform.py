from __future__ import annotations

from abc import ABC, abstractmethod

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
        if config.type != "identity":
            raise ValueError(f"Unsupported frame transform type: {config.type}")
        return cls(config.visible_to_agent)

    def transform_delta_position(self, delta_position: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(float(v) for v in delta_position)  # type: ignore[return-value]

    def transform_delta_rotation(self, delta_rotation_rpy: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(float(v) for v in delta_rotation_rpy)  # type: ignore[return-value]

    def metadata(self) -> dict[str, object]:
        return {"type": "identity", "visible_to_agent": self.visible_to_agent}
