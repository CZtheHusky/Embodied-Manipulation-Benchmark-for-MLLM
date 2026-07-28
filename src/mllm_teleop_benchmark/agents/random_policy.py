from __future__ import annotations

import random

from mllm_teleop_benchmark.agents.base import Agent, AgentContext
from mllm_teleop_benchmark.core.types import AgentDecision, AgentTrace


class RandomPolicy(Agent):
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, context: AgentContext) -> AgentDecision:
        return AgentDecision(
            action=context.action_space.sample(self._rng).as_dict(),
            trace=AgentTrace(actor_type="random"),
        )

    def metadata(self) -> dict[str, object]:
        return {"type": "random"}
