from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.base import AgentContext


@dataclass(frozen=True)
class PromptPayload:
    instruction: str
    control_instruction: str
    action_schema: dict[str, Any]
    observation_metadata: dict[str, Any]
    history_length: int


class PromptBuilder:
    def build(self, context: AgentContext) -> PromptPayload:
        control_instruction = ControlInstructionBuilder().build(
            context.metadata.get("control_interface", {}),
            context.action_schema,
        )
        return PromptPayload(
            instruction=context.task_instruction,
            control_instruction=control_instruction.text,
            action_schema=context.action_schema,
            observation_metadata=context.observation.metadata,
            history_length=len(context.history),
        )
