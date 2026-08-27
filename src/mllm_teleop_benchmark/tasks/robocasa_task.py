from __future__ import annotations

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class RoboCasaGenericTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instruction = config.params.get(
            "instruction",
            "Complete the RoboCasa household manipulation task.",
        )

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        if self.config.params.get("instruction_from_observation", False):
            key = self.config.params.get(
                "instruction_observation_key",
                "annotation.human.task_description",
            )
            value = observation.state.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    "RoboCasa reset observation is missing the required public "
                    f"task instruction {key!r}"
                )
            self._instruction = value.strip()
        else:
            self._instruction = self.config.params.get(
                "instruction",
                "Complete the RoboCasa household manipulation task.",
            )
        # Reset-success seeds would count as model successes without an agent command.
        if (
            self.config.params.get("reject_initial_success", False)
            and _success_from_state(sim.get_state())
        ):
            raise RuntimeError(
                "RoboCasa task is already successful immediately after reset "
                f"for seed {seed}; choose an explicit nontrivial seed"
            )
        return observation

    def instruction(self) -> str:
        return self._instruction

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        return _success_from_state(sim_state)

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        return {
            "is_success": _success_from_state(sim_state),
            "reward": sim_state.get("reward", 0.0),
        }


def _success_from_state(sim_state: dict[str, object]) -> bool:
    if "is_success" in sim_state:
        return bool(sim_state["is_success"])
    if "success" in sim_state:
        return bool(sim_state["success"])
    info = sim_state.get("info", {})
    if isinstance(info, dict):
        return bool(info.get("success", False))
    return False
