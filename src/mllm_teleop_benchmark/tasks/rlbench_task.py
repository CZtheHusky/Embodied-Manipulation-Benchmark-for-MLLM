from __future__ import annotations

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class RLBenchTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instruction = self.config.params.get(
            "instruction",
            "Move the robot end-effector to the target.",
        )

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        if self.config.params.get("instruction_from_observation", False):
            self._instruction = self._instruction_from_observation(observation)
        else:
            self._instruction = self.config.params.get(
                "instruction",
                "Move the robot end-effector to the target.",
            )
        # Reset-success seeds would count as model successes without an agent command.
        if (
            self.config.params.get("reject_initial_success", False)
            and self.is_success(observation, observation.state)
        ):
            raise RuntimeError(
                "RLBench task is already successful immediately after reset "
                f"for seed {seed}; choose an explicit nontrivial seed"
            )
        return observation

    def instruction(self) -> str:
        return self._instruction

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        success_key = self.config.params.get("success_key", "success")
        return sim_state.get(success_key, False)

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        success_key = self.config.params.get("success_key", "success")
        metrics: dict[str, object] = {
            "success": sim_state.get(success_key, False),
            "reward": sim_state.get("reward"),
            "terminate": sim_state.get("terminate", False),
        }
        for key in ("task_class", "variation_index", "step_id"):
            if key in sim_state:
                metrics[key] = sim_state[key]
        return metrics

    def _instruction_from_observation(self, observation: Observation) -> str:
        key = self.config.params.get(
            "instruction_observation_key",
            "descriptions",
        )
        descriptions = observation.state[key]
        index = self.config.params.get("instruction_description_index", 0)
        return descriptions[index].strip()
