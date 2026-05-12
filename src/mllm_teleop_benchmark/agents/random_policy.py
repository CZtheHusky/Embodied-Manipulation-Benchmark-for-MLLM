from __future__ import annotations

import random
from typing import Any

from mllm_teleop_benchmark.agents.base import Agent, AgentContext


class RandomPolicy(Agent):
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, context: AgentContext) -> Any:
        return context.action_space.sample(self._rng).as_dict()

    def metadata(self) -> dict[str, object]:
        return {"type": "random"}
