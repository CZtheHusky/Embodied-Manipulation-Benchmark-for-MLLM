from __future__ import annotations

from abc import ABC, abstractmethod

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class ObservationProvider(ABC):
    def configure_sim(self, sim: SimAdapter) -> None:
        return None

    @abstractmethod
    def get_observation(self, sim: SimAdapter) -> Observation:
        raise NotImplementedError


class DefaultObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def get_observation(self, sim: SimAdapter) -> Observation:
        state = sim.get_state()
        images = {camera.name: sim.render(camera) for camera in self.camera_manager.get_active_cameras()}
        return Observation(
            step_id=int(state.get("step_id", 0)),
            images=images,
            state=state,
            metadata={"camera": self.camera_manager.metadata()},
        )
