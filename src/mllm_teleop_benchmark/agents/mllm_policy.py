from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.agents.base import Agent, AgentContext


class MLLMPolicy(Agent):
    """Placeholder for future API-backed MLLM policies."""

    def act(self, context: AgentContext) -> Any:
        raise NotImplementedError("MLLMPolicy is an interface placeholder in v0")

    def metadata(self) -> dict[str, object]:
        return {"type": "mllm", "implemented": False}
