from __future__ import annotations

from abc import ABC, abstractmethod

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class Task(ABC):
    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    @abstractmethod
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def instruction(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError
