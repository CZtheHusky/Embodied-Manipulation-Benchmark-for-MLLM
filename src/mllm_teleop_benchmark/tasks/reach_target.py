from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class ReachTargetTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        return "Move the robot end effector to the target position."

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        return float(sim_state["distance_to_target"]) <= self.config.success_threshold

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        return {"final_distance_to_target": float(sim_state["distance_to_target"])}
