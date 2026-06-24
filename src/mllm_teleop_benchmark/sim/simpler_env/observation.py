from __future__ import annotations

from typing import Any

import numpy as np

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.simpler_env.adapter import SimplerEnvSimAdapter


class SimplerEnvObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def get_observation(self, sim: SimAdapter) -> Observation:
        if not isinstance(sim, SimplerEnvSimAdapter):
            raise TypeError(f"SimplerEnvObservationProvider requires SimplerEnvSimAdapter, got {type(sim).__name__}")
        state = sim.get_state()
        images = self._extract_images(sim.last_native_obs)
        if not images:
            frame = sim.extract_image()
            if frame is not None:
                images[self._default_camera_name()] = frame
        if not images:
            images = {camera.name: sim.render(camera) for camera in self.camera_manager.get_active_cameras()}
        return Observation(
            step_id=int(state.get("step_id", 0)),
            images=images,
            state=state,
            metadata={"camera": self.camera_manager.metadata(), "simulator": "simpler_env"},
        )

    def _default_camera_name(self) -> str:
        cameras = self.camera_manager.get_active_cameras()
        return cameras[0].name if cameras else "image"

    def _extract_images(self, observation: Any) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        self._collect_images(observation, images, prefix="")
        return images

    def _collect_images(self, value: Any, images: dict[str, np.ndarray], prefix: str) -> None:
        frame = self._as_hwc_uint8(value)
        if frame is not None:
            images[prefix or self._default_camera_name()] = frame
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            name = str(key)
            next_prefix = name if not prefix else f"{prefix}/{name}"
            if name in {"rgb", "image", "Color", "color"}:
                frame = self._as_hwc_uint8(item)
                if frame is not None:
                    images[prefix or self._default_camera_name()] = frame
                    continue
            self._collect_images(item, images, next_prefix)

    def _as_hwc_uint8(self, value: Any) -> np.ndarray | None:
        array = self._to_numpy(value)
        if array is None:
            return None
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            return None
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            return None
        if array.dtype != np.uint8:
            if float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.cpu().numpy()
        return None
