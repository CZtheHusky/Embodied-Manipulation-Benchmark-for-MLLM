from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class ManiSkillTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        if self.config.params.get("reject_initial_success", False):
            success_key = str(self.config.params.get("success_key", "success"))
            if bool(observation.state.get(success_key, False)):
                raise RuntimeError(
                    "ManiSkill task is already successful immediately after reset "
                    f"for seed {seed}; choose an explicit nontrivial seed"
                )
        return observation

    def instruction(self) -> str:
        return str(self.config.params.get("instruction", "Complete the ManiSkill manipulation task."))

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        success_key = str(self.config.params.get("success_key", "success"))
        return bool(sim_state.get(success_key, False))

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        success_key = str(self.config.params.get("success_key", "success"))
        metrics: dict[str, object] = {
            "success": bool(sim_state.get(success_key, False)),
            "reward": sim_state.get("reward"),
        }
        for key, value in sim_state.items():
            if isinstance(value, bool | int | float | str) and key not in metrics:
                metrics[key] = value
        return metrics
