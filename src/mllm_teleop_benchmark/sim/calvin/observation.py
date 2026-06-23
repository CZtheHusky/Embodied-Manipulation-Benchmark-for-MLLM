from __future__ import annotations

from typing import Any

import numpy as np

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.calvin.adapter import CalvinSimAdapter


class CalvinObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def configure_sim(self, sim: SimAdapter) -> None:
        if not isinstance(sim, CalvinSimAdapter):
            raise TypeError(f"CalvinObservationProvider requires CalvinSimAdapter, got {type(sim).__name__}")

        camera_configs: dict[str, dict[str, Any]] = {}
        for camera in self.camera_manager.get_active_cameras():
            config = {
                "name": camera.name,
                "width": camera.width,
                "height": camera.height,
            }
            if "position" in camera.params:
                config["position"] = camera.params["position"]
            if "look_at" in camera.params:
                config["look_at"] = camera.params["look_at"]
            camera_configs[camera.name] = config

        sim.set_camera_configuration(camera_configs)

    def get_observation(self, sim: SimAdapter) -> Observation:
        if not isinstance(sim, CalvinSimAdapter):
            raise TypeError(f"CalvinObservationProvider requires CalvinSimAdapter, got {type(sim).__name__}")

        state = sim.get_state()
        images = self._extract_images(sim.last_native_obs)

        if not images:
            for camera in self.camera_manager.get_active_cameras():
                images[camera.name] = sim.render(camera)

        return Observation(
            step_id=int(state.get("step_id", 0)),
            images=images,
            state=state,
            metadata={"camera": self.camera_manager.metadata(), "simulator": "calvin"},
        )

    def _extract_images(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return {}

        images: dict[str, Any] = {}
        rgb_obs = observation.get("rgb_obs", {})
        if not isinstance(rgb_obs, dict):
            return {}

        for key, payload in rgb_obs.items():
            camera_name = key.replace("rgb_", "")
            frame = self._as_hwc_uint8(payload)
            if frame is not None:
                images[camera_name] = frame

        return images

    def _as_hwc_uint8(self, value: Any) -> np.ndarray | None:
        if not isinstance(value, np.ndarray):
            return None

        array = value
        if array.ndim != 3:
            return None
        if array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.moveaxis(array, 0, -1)

        if array.dtype != np.uint8:
            if float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return array
