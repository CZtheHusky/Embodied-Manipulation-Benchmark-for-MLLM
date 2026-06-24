from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.libero.adapter import LiberoSimAdapter


class LiberoObservationProvider(ObservationProvider):
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

    def configure_sim(self, sim: SimAdapter) -> None:
        if not isinstance(sim, LiberoSimAdapter):
            raise TypeError(f"LiberoObservationProvider requires LiberoSimAdapter, got {type(sim).__name__}")
        sim.set_camera_configuration(self.camera_manager.get_active_cameras())

    def get_observation(self, sim: SimAdapter) -> Observation:
        if not isinstance(sim, LiberoSimAdapter):
            raise TypeError(f"LiberoObservationProvider requires LiberoSimAdapter, got {type(sim).__name__}")
        state = sim.get_state()
        images = sim._extract_images(sim.last_native_obs)
        if not images:
            images = {camera.name: sim.render(camera) for camera in self.camera_manager.get_active_cameras()}
        return Observation(
            step_id=int(state.get("step_id", 0)),
            images=images,
            state=state,
            metadata={
                "camera": self.camera_manager.metadata(),
                "simulator": "libero",
                "benchmark_validity": state.get("benchmark_validity", {}),
            },
        )
