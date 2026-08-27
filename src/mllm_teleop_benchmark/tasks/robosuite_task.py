from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class RobosuiteLiftTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
        # Reset-success seeds would count as model successes without an agent command.
        if (
            self.config.params.get("reject_initial_success", False)
            and self.is_success(observation, sim.get_state())
        ):
            raise RuntimeError(
                "Robosuite Lift task is already successful immediately after "
                f"reset for seed {seed}"
            )
        return observation

    def instruction(self) -> str:
        return "Lift the cube using the robot arm."

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        return sim_state.get("is_success", False)

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        return {
            "is_success": self.is_success(None, sim_state),
            "eef_pos": sim_state.get("eef_pos"),
        }
