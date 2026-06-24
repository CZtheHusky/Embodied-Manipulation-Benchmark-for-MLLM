from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class MetaworldTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        return str(self.config.params.get("instruction", "Complete the Meta-World manipulation task."))

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        success_key = str(self.config.params.get("success_key", "success"))
        if success_key in sim_state:
            value = sim_state[success_key]
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value) >= self.config.success_threshold
            return bool(value)
        reward = sim_state.get("reward")
        return reward is not None and float(reward) >= self.config.success_threshold

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        success_key = str(self.config.params.get("success_key", "success"))
        metrics: dict[str, object] = {
            "success": self.is_success(Observation(0, {}, {}, {}), sim_state),
            "reward": sim_state.get("reward"),
        }
        if success_key in sim_state and success_key not in metrics:
            metrics[success_key] = sim_state[success_key]
        for key, value in sim_state.items():
            if isinstance(value, bool | int | float | str) and key not in metrics:
                metrics[key] = value
        return metrics
