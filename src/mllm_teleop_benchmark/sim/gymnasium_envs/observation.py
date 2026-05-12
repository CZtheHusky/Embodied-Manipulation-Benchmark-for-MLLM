from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager, RandomizedMujocoCameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.gymnasium_envs.adapter import GymnasiumSimAdapter


class GymnasiumObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def get_observation(self, sim: SimAdapter) -> Observation:
        if not isinstance(sim, GymnasiumSimAdapter):
            raise TypeError(f"GymnasiumObservationProvider requires GymnasiumSimAdapter, got {type(sim).__name__}")
        if isinstance(self.camera_manager, RandomizedMujocoCameraManager):
            sim.set_camera_configuration(self.camera_manager.mujoco_camera_config())
        state = sim.get_state()
        images: dict[str, Any] = {}
        for camera in self.camera_manager.get_active_cameras():
            images[camera.name] = sim.render(camera)
        return Observation(
            step_id=int(state.get("step_id", 0)),
            images=images,
            state=state,
            metadata={
                "camera": self.camera_manager.metadata(),
                "simulator": "gymnasium",
                "env_id": state.get("env_id"),
            },
        )
