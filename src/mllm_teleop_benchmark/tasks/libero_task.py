from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class LiberoTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        # Reset-success seeds would count as model successes without an agent command.
        if (
            self.config.params.get("reject_initial_success", False)
            and self.is_success(observation, observation.state)
        ):
            raise RuntimeError(
                "LIBERO task is already successful immediately after reset "
                f"for seed {seed}; choose an explicit nontrivial seed"
            )
        return observation

    def instruction(self) -> str:
        if self.config.params.get("no_language_baseline", False):
            return ""
        return self.config.params.get(
            "language_instruction",
            self.config.params.get("instruction", "Complete the LIBERO manipulation task."),
        )

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        return sim_state.get("is_success", sim_state.get("success", False))

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        validity = dict(self.config.params.get("benchmark_validity", {}))
        runtime_validity = sim_state.get("benchmark_validity")
        if isinstance(runtime_validity, dict):
            validity.update(runtime_validity)

        return {
            "is_success": sim_state.get("is_success", False),
            "reward": sim_state.get("reward"),
            "benchmark_validity": validity,
        }
