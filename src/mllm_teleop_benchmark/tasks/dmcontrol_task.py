from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class DMControlTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        return str(self.config.params.get("instruction", "Complete the DMControl task."))

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        # DMControl tasks often return a reward between 0 and 1.
        # We can define success as reaching a certain reward threshold
        # or use a specific success key if the environment provides one.
        success_key = self.config.params.get("success_key")
        if success_key and success_key in sim_state:
            value = sim_state[success_key]
            if value is None:
                return False
            if success_key == "reward":
                return float(value) >= self.config.success_threshold
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value) >= self.config.success_threshold
            return bool(value)

        reward = sim_state.get("reward")
        if reward is not None:
            return float(reward) >= self.config.success_threshold

        return False

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        metrics: dict[str, object] = {
            "reward": sim_state.get("reward"),
        }

        # Add all scalar values from sim_state to metrics
        for key, value in sim_state.items():
            if isinstance(value, bool | int | float | str) and key not in metrics:
                metrics[key] = value

        # Determine success for metrics
        metrics["success"] = self.is_success(Observation(0, {}, {}, {}), sim_state)

        return metrics
