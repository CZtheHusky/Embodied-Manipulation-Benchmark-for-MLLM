from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mllm_teleop_benchmark.agents.base import AgentContext


@dataclass(frozen=True)
class PromptPayload:
    instruction: str
    action_schema: dict[str, Any]
    observation_metadata: dict[str, Any]
    history_length: int


class PromptBuilder:
    def build(self, context: AgentContext) -> PromptPayload:
        return PromptPayload(
            instruction=context.task_instruction,
            action_schema=context.action_schema,
            observation_metadata=context.observation.metadata,
            history_length=len(context.history),
        )
