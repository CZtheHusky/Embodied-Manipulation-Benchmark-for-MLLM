from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mllm_teleop_benchmark.config.schema import CameraSpec, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, SimulatorAction, StepResult


class SimAdapter(ABC):
    @abstractmethod
    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: SimulatorAction) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def render(self, camera: CameraSpec) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError
