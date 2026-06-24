from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class LiberoTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        if bool(self.config.params.get("no_language_baseline", False)):
            return ""
        return str(
            self.config.params.get(
                "language_instruction",
                self.config.params.get("instruction", "Complete the LIBERO manipulation task."),
            )
        )

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        return bool(sim_state.get("is_success", sim_state.get("success", False)))

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        validity = dict(self.config.params.get("benchmark_validity", {}))
        runtime_validity = sim_state.get("benchmark_validity")
        if isinstance(runtime_validity, dict):
            validity.update(runtime_validity)

        return {
            "is_success": bool(sim_state.get("is_success", False)),
            "reward": sim_state.get("reward"),
            "benchmark_validity": validity,
        }
