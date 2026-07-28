from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.core.types import AgentDecision, Observation, StepTransition


@dataclass(frozen=True)
class AgentContext:
    observation: Observation
    task_instruction: str
    action_space: ActionSpace
    action_schema: dict[str, Any]
    metadata: dict[str, Any]
    history: tuple[StepTransition, ...] = ()


class Agent(ABC):
    @abstractmethod
    def act(self, context: AgentContext) -> AgentDecision:
        raise NotImplementedError

    def metadata(self) -> dict[str, object]:
        return {"type": self.__class__.__name__}
