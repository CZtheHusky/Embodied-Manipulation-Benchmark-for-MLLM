from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class GymnasiumReacherTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        return "Move the Reacher fingertip to the target sphere."

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        distance = sim_state.get("distance_to_target", sim_state.get("goal_distance"))
        if distance is None:
            return False
        return float(distance) <= self.config.success_threshold

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        distance = sim_state.get("distance_to_target", sim_state.get("goal_distance", float("inf")))
        return {
            "final_distance_to_target": float(distance),
            "goal_hold_counter": int(sim_state.get("goal_hold_counter", 0)),
        }
