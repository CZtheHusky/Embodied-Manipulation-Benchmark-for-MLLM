from __future__ import annotations

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class LanguageTableTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self.current_instruction: str | None = None

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        self.current_instruction = str(observation.state["instruction"])
        return observation

    def instruction(self) -> str:
        if self.current_instruction is not None:
            return self.current_instruction
        return str(
            self.config.params.get(
                "instruction", "Complete the current Language-Table instruction."
            )
        )

    def is_success(
        self,
        observation: Observation,
        sim_state: dict[str, object],
    ) -> bool:
        return bool(sim_state.get("success", False))

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in sim_state.items()
            if key
            in {
                "reward",
                "success",
                "instruction",
                "object_target_distance",
                "distance_to_target",
            }
        }
