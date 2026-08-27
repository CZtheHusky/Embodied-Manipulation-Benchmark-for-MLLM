from __future__ import annotations

from typing import Any

import numpy as np

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class RLBenchObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def configure_sim(self, sim: SimAdapter) -> None:
        sim.set_camera_configuration(self.camera_manager.get_active_cameras())

    def get_observation(self, sim: SimAdapter) -> Observation:
        state = sim.get_state()
        images = self._extract_images(sim.last_native_obs)
        if not images:
            images = {camera.name: sim.render(camera) for camera in self.camera_manager.get_active_cameras()}
        return Observation(
            step_id=state.get("step_id", 0),
            images=images,
            state=state,
            metadata={
                "camera": self.camera_manager.metadata(),
                "simulator": sim.config.type,
            },
        )

    def _extract_images(self, observation: Any) -> dict[str, np.ndarray]:
        if observation is None:
            return {}
        images: dict[str, np.ndarray] = {}
        for camera in self.camera_manager.get_active_cameras():
            for attr in (camera.name, f"{camera.name}_rgb"):
                frame = self._as_hwc_uint8(getattr(observation, attr, None))
                if frame is not None:
                    images[camera.name] = frame
                    break
        return images

    def _as_hwc_uint8(self, value: Any) -> np.ndarray | None:
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            return None
        array = value
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
