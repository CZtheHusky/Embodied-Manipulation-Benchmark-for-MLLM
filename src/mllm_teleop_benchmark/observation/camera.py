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
        self.config = config
        self._rng = random.Random()
        self._active = config.cameras
        self._sampled_config: dict[str, object] | None = None

    def reset(self, seed: int) -> None:
        self._rng.seed(seed)
        self._active = self.config.cameras
        ranges = self.config.params
        self._sampled_config = {
            "distance": self._rng.uniform(*map(float, ranges.get("distance_range", [0.7, 2.0]))),
            "azimuth": self._rng.uniform(*map(float, ranges.get("azimuth_range", [0.0, 90.0]))),
            "elevation": self._rng.uniform(*map(float, ranges.get("elevation_range", [-90.0, -20.0]))),
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

class RandomizedManiSkillCameraManager(CameraManager):
    """Episode-level ManiSkill camera sampler.

    Camera positions are sampled once per episode from each camera's
    params.position_ranges. Orientation stays fixed from pose/q params.
    """

    def __init__(self, config: CameraConfig) -> None:
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
                position = tuple(self._rng.uniform(*map(float, value)) for value in ranges)
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
        return tuple(float(v) for v in quat)

class RandomizedCalvinCameraManager(CameraManager):
    """Episode-level CALVIN camera sampler."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._rng = random.Random()
        self._active = config.cameras

    def reset(self, seed: int) -> None:
        self._rng.seed(seed)
        active: list[CameraSpec] = []
        for camera in self.config.cameras:
            params = dict(camera.params)
            ranges = params.pop("position_ranges", None)
            look_at_ranges = params.pop("look_at_ranges", None)
            if ranges is not None:
                params["position"] = tuple(self._rng.uniform(*map(float, value)) for value in ranges)
            if look_at_ranges is not None:
                params["look_at"] = tuple(self._rng.uniform(*map(float, value)) for value in look_at_ranges)
            active.append(
                CameraSpec(
                    name=camera.name,
                    type=camera.type,
                    width=camera.width,
                    height=camera.height,
                    pose=camera.pose,
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
                {
                    "name": camera.name,
                    "type": camera.type,
                    "width": camera.width,
                    "height": camera.height,
                    "pose": camera.pose,
                    "params": dict(camera.params),
                }
                for camera in self._active
            ],
        }
