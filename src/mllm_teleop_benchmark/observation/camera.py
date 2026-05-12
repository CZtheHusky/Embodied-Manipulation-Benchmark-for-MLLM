from __future__ import annotations

from abc import ABC, abstractmethod
import random

from mllm_teleop_benchmark.config.schema import CameraConfig, CameraSpec


class CameraManager(ABC):
    @abstractmethod
    def reset(self, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_active_cameras(self) -> tuple[CameraSpec, ...]:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, object]:
        raise NotImplementedError


class FixedCameraManager(CameraManager):
    def __init__(self, config: CameraConfig) -> None:
        if config.manager != "fixed":
            raise ValueError(f"Unsupported camera manager: {config.manager}")
        self.config = config

    def reset(self, seed: int) -> None:
        return None

    def get_active_cameras(self) -> tuple[CameraSpec, ...]:
        return self.config.cameras

    def metadata(self) -> dict[str, object]:
        if not self.config.visible_metadata:
            return {"manager": self.config.manager, "visible_metadata": False}
        return {
            "manager": self.config.manager,
            "visible_metadata": True,
            "cameras": [
                {"name": camera.name, "type": camera.type, "width": camera.width, "height": camera.height, "pose": camera.pose}
                for camera in self.config.cameras
            ],
        }


class RandomizedMujocoCameraManager(CameraManager):
    """Episode-level MuJoCo free-camera sampler."""

    def __init__(self, config: CameraConfig) -> None:
        if config.manager != "mujoco_randomized":
            raise ValueError(f"Unsupported camera manager: {config.manager}")
        self.config = config
        self._rng = random.Random()
        self._active = config.cameras
        self._sampled_config: dict[str, object] | None = None

    def reset(self, seed: int) -> None:
        self._rng.seed(seed)
        self._active = self.config.cameras
        ranges = self.config.params
        self._sampled_config = {
            "distance": self._uniform_range(ranges.get("distance_range", [0.7, 2.0])),
            "azimuth": self._uniform_range(ranges.get("azimuth_range", [0.0, 90.0])),
            "elevation": self._uniform_range(ranges.get("elevation_range", [-90.0, -20.0])),
            "lookat": tuple(float(v) for v in ranges.get("lookat", [0.0, 0.0, 0.0])),
        }

    def get_active_cameras(self) -> tuple[CameraSpec, ...]:
        return self._active

    def metadata(self) -> dict[str, object]:
        if not self.config.visible_metadata:
            return {"manager": self.config.manager, "visible_metadata": False}
        return {
            "manager": self.config.manager,
            "visible_metadata": True,
            "mujoco_camera_config": dict(self._sampled_config or {}),
        }

    def mujoco_camera_config(self) -> dict[str, object] | None:
        return dict(self._sampled_config) if self._sampled_config is not None else None

    def _uniform_range(self, value: object) -> float:
        if not isinstance(value, list | tuple) or len(value) != 2:
            raise ValueError(f"Camera range must be [low, high], got {value!r}")
        return self._rng.uniform(float(value[0]), float(value[1]))
