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


class RandomizedManiSkillCameraManager(CameraManager):
    """Episode-level ManiSkill camera sampler.

    Camera positions are sampled once per episode from each camera's
    params.position_ranges. Orientation stays fixed from pose/q params.
    """

    def __init__(self, config: CameraConfig) -> None:
        if config.manager != "maniskill_randomized":
            raise ValueError(f"Unsupported camera manager: {config.manager}")
        self.config = config
        self._rng = random.Random()
        self._active = config.cameras

    def reset(self, seed: int) -> None:
        self._rng.seed(seed)
        active: list[CameraSpec] = []
        for camera in self.config.cameras:
            params = dict(camera.params)
            ranges = params.pop("position_ranges", None)
            pose = camera.pose
            if ranges is not None:
                position = tuple(self._uniform_range(value) for value in ranges)
                quat = self._camera_quat(camera)
                pose = (*position, *quat)
            active.append(
                CameraSpec(
                    name=camera.name,
                    type=camera.type,
                    width=camera.width,
                    height=camera.height,
                    pose=pose,
                    params=params,
                )
            )
        self._active = tuple(active)

    def get_active_cameras(self) -> tuple[CameraSpec, ...]:
        return self._active

    def metadata(self) -> dict[str, object]:
        if not self.config.visible_metadata:
            return {"manager": self.config.manager, "visible_metadata": False}
        return {
            "manager": self.config.manager,
            "visible_metadata": True,
            "cameras": [
                {"name": camera.name, "type": camera.type, "width": camera.width, "height": camera.height, "pose": camera.pose}
                for camera in self._active
            ],
        }

    def _camera_quat(self, camera: CameraSpec) -> tuple[float, float, float, float]:
        if len(camera.pose) == 7:
            return tuple(float(v) for v in camera.pose[3:7])
        quat = camera.params.get("q", camera.params.get("quat", (1.0, 0.0, 0.0, 0.0)))
        if not isinstance(quat, list | tuple) or len(quat) != 4:
            raise ValueError(f"ManiSkill camera {camera.name} quaternion must contain 4 values")
        return tuple(float(v) for v in quat)

    def _uniform_range(self, value: object) -> float:
        if not isinstance(value, list | tuple) or len(value) != 2:
            raise ValueError(f"Camera position range must be [low, high], got {value!r}")
        return self._rng.uniform(float(value[0]), float(value[1]))
