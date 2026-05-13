from __future__ import annotations

from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class GymnasiumPusherTask(Task):
    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        return "Move the white cylinder to the red goal disk using the robot end effector."

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        distance = sim_state.get("object_goal_distance", sim_state.get("distance_to_target"))
        if distance is None:
            return False
        return float(distance) <= self.config.success_threshold

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        object_goal = sim_state.get("object_goal_distance", sim_state.get("distance_to_target", float("inf")))
        fingertip_object = sim_state.get("fingertip_object_distance", float("inf"))
        return {
            "final_object_goal_distance": float(object_goal),
            "final_fingertip_object_distance": float(fingertip_object),
            "goal_hold_counter": int(sim_state.get("goal_hold_counter", 0)),
        }
