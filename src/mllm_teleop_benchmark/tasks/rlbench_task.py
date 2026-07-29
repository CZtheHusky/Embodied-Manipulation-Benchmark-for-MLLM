from __future__ import annotations

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class RLBenchTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instruction = self._configured_instruction()

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        if self.config.params.get("instruction_from_observation", False):
            self._instruction = self._instruction_from_observation(observation)
        else:
            self._instruction = self._configured_instruction()
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
        success_key = str(self.config.params.get("success_key", "success"))
        return bool(sim_state.get(success_key, False))

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        success_key = str(self.config.params.get("success_key", "success"))
        metrics: dict[str, object] = {
            "success": bool(sim_state.get(success_key, False)),
            "reward": sim_state.get("reward"),
            "terminate": bool(sim_state.get("terminate", False)),
        }
        for key in ("task_class", "variation_index", "step_id"):
            if key in sim_state:
                metrics[key] = sim_state[key]
        return metrics

    def _configured_instruction(self) -> str:
        return str(
            self.config.params.get(
                "instruction",
                "Move the robot end-effector to the target.",
            )
        )

    def _instruction_from_observation(self, observation: Observation) -> str:
        key = str(
            self.config.params.get(
                "instruction_observation_key",
                "descriptions",
            )
        )
        descriptions = observation.state.get(key)
        if not isinstance(descriptions, list | tuple) or not descriptions:
            raise RuntimeError(
                "RLBench reset observation is missing the required public "
                f"task descriptions {key!r}"
            )
        index = int(self.config.params.get("instruction_description_index", 0))
        if index < 0 or index >= len(descriptions):
            raise RuntimeError(
                "RLBench public task description index "
                f"{index} is out of range for {len(descriptions)} descriptions"
            )
        instruction = descriptions[index]
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError(
                "RLBench reset observation contains an invalid public task "
                f"description at index {index}"
            )
        return instruction.strip()
