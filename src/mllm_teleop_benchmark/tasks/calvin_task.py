from __future__ import annotations

from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class CalvinTask(Task):
    _DRAWER_JOINT = "base__drawer"
    _DRAWER_CLOSED_STATE = 0.0
    _DRAWER_CLOSED_ATOL = 1e-6

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._target_color = config.params.get("target_block_color", "red").strip().lower()
        self._instruction = config.params.get(
            "instruction",
            (
                f"Open the drawer, pick up the {self._target_color} block, "
                f"and place the {self._target_color} block inside the drawer."
            ),
        )
        self._task_oracle: Any | None = None
        self._reset_info: dict[str, object] | None = None
        self._opened_info: dict[str, object] | None = None
        self._held_info: dict[str, object] | None = None
        self._success = False

    def reset(self, sim: SimAdapter, seed: int) -> Observation:
        self._reset_info = None
        self._opened_info = None
        self._held_info = None
        self._success = False
        observation = sim.reset(self.config, seed)
        self._reset_info = self._copy_oracle_info(observation.state)
        if self.config.params.get("reject_initial_success", False):
            violations = self._reset_invariant_violations(self._reset_info)
            # A violated reset invariant would make the staged task trivially or incorrectly solved.
            if violations:
                raise RuntimeError(
                    f"CALVIN reset invariant failed for seed {seed}: "
                    + "; ".join(violations)
                )
        if self._task_oracle is None:
            self._task_oracle = self._load_official_task_oracle()
        return observation

    def instruction(self) -> str:
        return self._instruction

    def is_success(self, observation: Observation, sim_state: dict[str, object]) -> bool:
        if self._success:
            return True
        # reset() initializes both objects before EpisodeManager can query success.
        assert self._reset_info is not None and self._task_oracle is not None

        current_info = self._copy_oracle_info(sim_state)
        if self._opened_info is None:
            achieved = self._task_oracle.get_task_info_for_set(
                self._reset_info,
                current_info,
                {"open_drawer"},
            )
            if "open_drawer" in achieved:
                self._opened_info = deepcopy(current_info)
            return False

        if self._held_info is None:
            achieved = self._task_oracle.get_task_info_for_set(
                self._opened_info,
                current_info,
                {
                    f"lift_{self._target_color}_block_table",
                    f"lift_{self._target_color}_block_slider",
                },
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

    @staticmethod
    def _copy_oracle_info(state: dict[str, object]) -> dict[str, object]:
        return deepcopy(
            {
                "robot_info": state["robot_info"],
                "scene_info": state["scene_info"],
            }
        )

    def _reset_invariant_violations(
        self, info: dict[str, object]
    ) -> tuple[str, ...]:
        try:
            robot_info = info["robot_info"]
            scene_info = info["scene_info"]
            drawer_state = scene_info["doors"][self._DRAWER_JOINT]["current_state"]
            table_info = scene_info["fixed_objects"]["table"]
            target_info = scene_info["movable_objects"][
                f"block_{self._target_color}"
            ]
            robot_uid = robot_info["uid"]
            target_uid = target_info["uid"]
            drawer_contact = (
                table_info["uid"],
                table_info["links"]["drawer_link"],
            )
            robot_contact_uids = self._contact_body_uids(robot_info["contacts"])
            target_contact_uids = self._contact_body_uids(target_info["contacts"])
            target_contact_pairs = self._contact_body_links(target_info["contacts"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "CALVIN reset invariant requires pinned scene_info and robot_info fields"
            ) from exc

        if not isinstance(drawer_state, Real) or isinstance(drawer_state, bool):
            raise RuntimeError(
                "CALVIN reset invariant requires a numeric base__drawer current_state"
            )

        violations = []
        if (
            abs(float(drawer_state) - self._DRAWER_CLOSED_STATE)
            > self._DRAWER_CLOSED_ATOL
        ):
            violations.append(f"drawer is not closed (state={float(drawer_state):.6g})")
        if drawer_contact in target_contact_pairs:
            violations.append(
                f"{self._target_color} block is in contact with the drawer link"
            )
        if target_uid in robot_contact_uids or robot_uid in target_contact_uids:
            violations.append(
                f"{self._target_color} block is in contact with the robot"
            )
        return tuple(violations)

    @staticmethod
    def _contact_body_uids(contacts: object) -> set[object]:
        try:
            return {contact[2] for contact in contacts}
        except (IndexError, TypeError) as exc:
            raise RuntimeError(
                "CALVIN reset invariant requires PyBullet contact tuples"
            ) from exc

    @staticmethod
    def _contact_body_links(contacts: object) -> set[tuple[object, object]]:
        try:
            return {(contact[2], contact[4]) for contact in contacts}
        except (IndexError, TypeError) as exc:
            raise RuntimeError(
                "CALVIN reset invariant requires PyBullet contact tuples"
            ) from exc

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
