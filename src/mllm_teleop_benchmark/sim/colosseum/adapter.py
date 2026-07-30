from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mllm_teleop_benchmark.config.schema import SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation
from mllm_teleop_benchmark.sim.coppeliasim_lock import (
    CoppeliaSimRuntimeBusyError,
)
from mllm_teleop_benchmark.sim.rlbench.adapter import (
    RLBenchSimAdapter,
    _load_invalid_action_error_type,
)


class ColosseumSimAdapter(RLBenchSimAdapter):
    simulator_type = "colosseum"

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self.task_name = str(config.env_id or config.params.get("task_name", "close_box"))
        self.spreadsheet_idx = int(config.params.get("spreadsheet_idx", 2))
        self.perturbation_name = str(
            config.params.get("perturbation", "manip_obj_color")
        )
        if self.task_name != "close_box":
            raise ValueError(
                "The minimal Colosseum backend supports only task close_box"
            )
        if self.spreadsheet_idx != 2 or self.perturbation_name != "manip_obj_color":
            raise ValueError(
                "The minimal Colosseum backend supports only "
                "spreadsheet_idx=2 (manip_obj_color)"
            )
        self.task_class_name = self.task_name
        self.episode_seed: int | None = None
        self._official_config: Any | None = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        if self.env is not None:
            self.close()
        self.episode_seed = int(seed)
        return super().reset(task_config, seed)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["evaluator_metadata"] = {
            "task": self.task_name,
            "perturbation": self.perturbation_name,
            "spreadsheet_idx": self.spreadsheet_idx,
            "variation": self.variation_index,
            "seed": self.episode_seed,
        }
        return state

    def _ensure_task(self, task_config: TaskConfig | None) -> None:
        if self.task is not None:
            return
        if self.episode_seed is None:
            raise RuntimeError("Colosseum reset requires an episode seed")
        try:
            import pyrep  # noqa: F401
            from colosseum import (
                ASSETS_CONFIGS_FOLDER,
                ASSETS_JSON_FOLDER,
                TASKS_PY_FOLDER,
                TASKS_TTM_FOLDER,
            )
            from colosseum.rlbench.extensions.environment import EnvironmentExt
            from colosseum.rlbench.utils import (
                get_spreadsheet_config,
                name_to_class,
            )
            from omegaconf import OmegaConf

            invalid_action_error_type = _load_invalid_action_error_type()
        except ImportError as exc:
            raise ImportError(
                "Colosseum backend requires colosseum, its pinned RLBench/PyRep "
                "sources, and an external CoppeliaSim 4.1 install. Create the "
                "backend with `python3 scripts/manage_conda_env.py create "
                "colosseum`, then configure the CoppeliaSim display/runtime."
            ) from exc

        base_config = OmegaConf.load(
            Path(ASSETS_CONFIGS_FOLDER) / f"{self.task_name}.yaml"
        )
        with (Path(ASSETS_JSON_FOLDER) / f"{self.task_name}.json").open(
            encoding="utf-8"
        ) as handle:
            collection_config = json.load(handle)
        strategy = collection_config.get("strategy", [])
        if self.spreadsheet_idx >= len(strategy):
            raise RuntimeError(
                f"Colosseum task {self.task_name} has no spreadsheet_idx="
                f"{self.spreadsheet_idx}"
            )
        selected = strategy[self.spreadsheet_idx]
        if (
            int(selected.get("spreadsheet_idx", -1)) != self.spreadsheet_idx
            or selected.get("variation_name") != self.perturbation_name
        ):
            raise RuntimeError(
                "Colosseum official strategy does not map spreadsheet_idx=2 "
                "to manip_obj_color"
            )
        base_config.env.seed = self.episode_seed
        self._official_config = get_spreadsheet_config(
            base_config,
            collection_config,
            self.spreadsheet_idx,
        )

        task_class = name_to_class(self.task_name, TASKS_PY_FOLDER)
        if task_class is None:
            raise RuntimeError(
                f"Colosseum could not resolve official task {self.task_name}"
            )
        self._invalid_action_error_type = invalid_action_error_type
        kwargs = dict(self.config.env_kwargs)
        kwargs.setdefault("headless", self.headless)
        kwargs.setdefault("obs_config", self._observation_config())
        kwargs.setdefault(
            "static_positions",
            bool(self.config.params.get("static_positions", False)),
        )
        kwargs.setdefault("path_task_ttms", TASKS_TTM_FOLDER)
        kwargs.setdefault("env_config", self._official_config.env)
        if "dataset_root" in self.config.params:
            kwargs.setdefault("dataset_root", self.config.params["dataset_root"])
        if "robot_setup" in self.config.params:
            kwargs.setdefault("robot_setup", self.config.params["robot_setup"])

        try:
            self.env = EnvironmentExt(action_mode=self._action_mode(), **kwargs)
            self._launch_environment(task_class)
        except CoppeliaSimRuntimeBusyError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Failed to launch Colosseum EnvironmentExt with CoppeliaSim "
                "4.1. Verify the pinned sources, CoppeliaSim environment "
                "variables, and DISPLAY/headless GPU setup."
            ) from exc

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={"simulator": self.simulator_type, "env_id": self.task_name},
        )
