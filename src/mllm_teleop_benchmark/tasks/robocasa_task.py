from __future__ import annotations

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class RoboCasaGenericTask(Task):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        return sim.reset(self.config, seed)

    def instruction(self) -> str:
        return str(self.config.params.get("instruction", "Complete the RoboCasa household manipulation task."))

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        return _success_from_state(sim_state)

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        return {
            "is_success": _success_from_state(sim_state),
            "reward": sim_state.get("reward", 0.0),
        }


def _success_from_state(sim_state: dict[str, object]) -> bool:
    if "is_success" in sim_state:
        return bool(sim_state["is_success"])
    if "success" in sim_state:
        return bool(sim_state["success"])
    info = sim_state.get("info", {})
    if isinstance(info, dict):
        return bool(info.get("success", False))
    return False
