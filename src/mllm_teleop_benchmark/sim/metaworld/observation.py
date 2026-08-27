from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class MetaworldObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def get_observation(self, sim: SimAdapter) -> Observation:
        state = sim.get_state()
        images: dict[str, Any] = {}
        for camera in self.camera_manager.get_active_cameras():
            images[camera.name] = sim.render(camera)
        return Observation(
            step_id=state.get("step_id", 0),
            images=images,
            state=state,
            metadata={
                "camera": self.camera_manager.metadata(),
                "simulator": "metaworld",
                "env_id": state.get("env_id"),
                "env_name": state.get("env_name"),
            },
        )
