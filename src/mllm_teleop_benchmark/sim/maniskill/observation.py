from __future__ import annotations

from typing import Any

import numpy as np

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.maniskill.adapter import ManiSkillSimAdapter
from mllm_teleop_benchmark.sim.maniskill.camera import maniskill_sensor_config


class ManiSkillObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def configure_sim(self, sim: SimAdapter) -> None:
        if not isinstance(sim, ManiSkillSimAdapter):
            raise TypeError(f"ManiSkillObservationProvider requires ManiSkillSimAdapter, got {type(sim).__name__}")
        sim.set_camera_configuration(maniskill_sensor_config(self.camera_manager.get_active_cameras()))

    def get_observation(self, sim: SimAdapter) -> Observation:
        if not isinstance(sim, ManiSkillSimAdapter):
            raise TypeError(f"ManiSkillObservationProvider requires ManiSkillSimAdapter, got {type(sim).__name__}")
        state = sim.get_state()
        images = self._extract_images(sim.last_native_obs)
        if not images:
            images = {camera.name: sim.render(camera) for camera in self.camera_manager.get_active_cameras()}
        return Observation(
            step_id=int(state.get("step_id", 0)),
            images=images,
            state=state,
            metadata={"camera": self.camera_manager.metadata(), "simulator": "maniskill"},
        )

    def _extract_images(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return {}
        sensor_data = observation.get("sensor_data", observation.get("image", observation.get("rgb", {})))
        if not isinstance(sensor_data, dict):
            return {}
        images: dict[str, Any] = {}
        for camera_name, payload in sensor_data.items():
            frame = self._camera_rgb(payload)
            if frame is not None:
                images[str(camera_name)] = frame
        return images

    def _camera_rgb(self, payload: Any) -> Any | None:
        if isinstance(payload, dict):
            for key in ("rgb", "Color", "color"):
                if key in payload:
                    return self._as_hwc_uint8(payload[key])
            return None
        return self._as_hwc_uint8(payload)

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
