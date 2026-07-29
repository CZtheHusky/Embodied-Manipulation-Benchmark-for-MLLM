from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class CalvinTask(Task):
    _BLOCK_COLORS = frozenset({"red", "blue", "pink"})

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._target_color = str(config.params.get("target_block_color", "red")).strip().lower()
        if self._target_color not in self._BLOCK_COLORS:
            raise ValueError(
                "task.params.target_block_color must be one of: blue, pink, red"
            )
        self._instruction = str(
            config.params.get(
                "instruction",
                (
                    f"Open the drawer, pick up the {self._target_color} block, "
                    f"and place the {self._target_color} block inside the drawer."
                ),
            )
        )
        self._task_oracle: Any | None = None
        self._reset_info: dict[str, object] | None = None
        self._held_info: dict[str, object] | None = None
        self._success = False

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        self._reset_info = None
        self._held_info = None
        self._success = False
        observation = sim.reset(self.config, seed)
        self._reset_info = self._copy_oracle_info(observation.state)
        if self._task_oracle is None:
            self._task_oracle = self._load_official_task_oracle()
        return observation

    def instruction(self) -> str:
        return self._instruction

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        if self._success:
            return True
        if self._reset_info is None or self._task_oracle is None:
            raise RuntimeError("CALVIN task success was checked before reset()")

        current_info = self._copy_oracle_info(sim_state)
        if self._held_info is None:
            achieved = self._task_oracle.get_task_info_for_set(
                self._reset_info,
                current_info,
                self._target_lift_tasks(),
            )
            if achieved:
                self._held_info = deepcopy(current_info)
            return False

        achieved = self._task_oracle.get_task_info_for_set(
            self._held_info,
            current_info,
            {"place_in_drawer"},
        )
        self._success = "place_in_drawer" in achieved
        return self._success

    def metrics(self, sim_state: dict[str, object]) -> dict[str, object]:
        metrics: dict[str, object] = {
            "success": self.is_success(Observation(0, {}, {}, {}), sim_state),
            "reward": sim_state.get("reward"),
        }
        for key, value in sim_state.items():
            if isinstance(value, (bool, int, float, str)) and key not in metrics:
                metrics[key] = value
        return metrics

    def _target_lift_tasks(self) -> set[str]:
        return {
            f"lift_{self._target_color}_block_table",
            f"lift_{self._target_color}_block_slider",
        }

    @staticmethod
    def _copy_oracle_info(state: dict[str, object]) -> dict[str, object]:
        missing = {"robot_info", "scene_info"} - state.keys()
        if missing:
            raise RuntimeError(
                "CALVIN task oracle requires reset/step info fields: "
                + ", ".join(sorted(missing))
            )
        return deepcopy(
            {
                "robot_info": state["robot_info"],
                "scene_info": state["scene_info"],
            }
        )

    @staticmethod
    def _load_official_task_oracle() -> Any:
        try:
            import calvin_env
            from calvin_env.envs.tasks import Tasks
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                "CALVIN task success requires calvin_env and omegaconf"
            ) from exc

        task_config_path = (
            Path(calvin_env.__file__).parents[1]
            / "conf"
            / "tasks"
            / "new_playtable_tasks.yaml"
        )
        task_config = OmegaConf.load(task_config_path)
        return Tasks(task_config.tasks)
