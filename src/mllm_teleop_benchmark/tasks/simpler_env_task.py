from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class SimplerEnvTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instruction = str(config.params.get("instruction", "Complete the SimplerEnv manipulation task."))

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        instruction = observation.state.get("language_instruction")
        if instruction is not None:
            self._instruction = str(instruction)
        if (
            self.config.params.get("reject_initial_success", False)
            and self.is_success(observation, sim.get_state())
        ):
            raise RuntimeError(
                "SimplerEnv task is already successful immediately after reset "
                f"for seed {seed}; choose an explicit nontrivial seed"
            )
        return observation

    def instruction(self) -> str:
        return self._instruction

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        success_key = str(self.config.params.get("success_key", "success"))
        value = self._get_nested(sim_state, success_key)
        if value is None:
            raise RuntimeError(
                f"SimplerEnv simulator state is missing success key {success_key!r}"
            )
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return float(value) >= self.config.success_threshold
        return bool(value)

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        success_key = str(self.config.params.get("success_key", "success"))
        metrics: dict[str, object] = {
            "success": self.is_success(None, sim_state),
            "reward": sim_state.get("reward"),
        }
        episode_stats = sim_state.get("episode_stats")
        if isinstance(episode_stats, dict):
            for key, value in episode_stats.items():
                if isinstance(value, bool | int | float | str):
                    metrics[f"episode_stats/{key}"] = value
        for key, value in sim_state.items():
            if isinstance(value, bool | int | float | str) and key not in metrics:
                metrics[key] = value
        if success_key not in metrics:
            metrics[success_key] = self._get_nested(sim_state, success_key)
        return metrics

    def _get_nested(self, data: dict[str, object], key_path: str) -> Any:
        cursor: Any = data
        for key in key_path.split("."):
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor
