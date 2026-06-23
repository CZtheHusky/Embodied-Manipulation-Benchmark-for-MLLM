from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class CalvinTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        # CALVIN environments often provide the current language instruction in the task info
        # But we can also specify it in the config
        return str(self.config.params.get("instruction", "Complete the CALVIN manipulation task."))

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        for key in self._success_keys():
            if key in sim_state:
                return bool(sim_state[key])
        return False

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        metrics: dict[str, object] = {
            "success": self.is_success(None, sim_state),
            "reward": sim_state.get("reward"),
        }
        # Add any other scalar metrics from sim_state
        for key, value in sim_state.items():
            if isinstance(value, (bool, int, float, str)) and key not in metrics:
                metrics[key] = value
        return metrics

    def _success_keys(self) -> tuple[str, ...]:
        value = self.config.params.get("success_keys", ("success", "is_success", "task_success"))
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)
