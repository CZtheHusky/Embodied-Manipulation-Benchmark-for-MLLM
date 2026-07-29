from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class RobosuiteLiftTask(Task):
    """
    A task wrapper for robosuite's 'Lift' environment.
    """

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        observation = sim.reset(self.config, seed)
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
        # robosuite environments typically return success in info or have a method.
        # Our adapter stores some state in sim_state, but for robosuite we can
        # also check the adapter's wrapped env directly if needed, or use the 
        # simulator state provided.
        # In robosuite, _check_success() is the standard way.
        # Since SimAdapter doesn't expose the raw env by default, we'll assume
        # the adapter provides enough info in get_state() or we can extend SimAdapter.
        # For now, we'll assume the adapter's get_state() might eventually include 
        # success if we update it, or we rely on the sim providing it in 'info'.
        # However, Task.is_success receives sim_state.
        return bool(sim_state.get("is_success", False))

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        return {
            "is_success": self.is_success(None, sim_state),
            "eef_pos": sim_state.get("eef_pos"),
        }
