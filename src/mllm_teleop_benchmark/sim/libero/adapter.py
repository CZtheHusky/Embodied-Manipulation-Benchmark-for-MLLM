from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class LiberoSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.env: Any | None = None
        self.step_id = 0
        self.last_native_obs: Any = None
        self.last_reward: float | None = None
        self.last_done = False
        self.last_info: dict[str, Any] = {}
        self._camera_specs: tuple[CameraSpec, ...] = ()
        self._task_metadata: dict[str, Any] = {}
        self._max_steps = 0

    def set_camera_configuration(self, camera_specs: tuple[CameraSpec, ...]) -> None:
        self._camera_specs = tuple(camera_specs)

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        benchmark, offscreen_env = self._libero_modules()
        self.close()
        self.step_id = 0
        self.last_reward = None
        self.last_done = False
        self.last_info = {}
        self._max_steps = self.config.params.get(
            "max_episode_steps", task_config.max_steps
        )

        suite_name = task_config.params.get(
            "suite_name", self.config.params.get("suite_name", "libero_spatial")
        )
        task_id = task_config.params.get(
            "task_id", self.config.params.get("task_id", 0)
        )
        task_order_index = task_config.params.get(
            "task_order_index", self.config.params.get("task_order_index", 0)
        )

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name](task_order_index=task_order_index)
        task = task_suite.get_task(task_id)
        self._validate_language_instruction(task_config, task)
        bddl_file = task_suite.get_task_bddl_file_path(task_id)

        init_states = task_suite.get_task_init_states(task_id)
        init_state_id, init_state_strategy = self._select_init_state(
            task_config,
            seed,
            len(init_states),
        )
        self._task_metadata = {
            "language_instruction": task.language,
            "suite_name": suite_name,
            "task_id": task_id,
            "task_name": task.name,
            "init_state_id": init_state_id,
            "init_state_strategy": init_state_strategy,
            "settling_steps": self.config.params.get("settling_steps", 5),
            "no_language_baseline": task_config.params.get(
                "no_language_baseline", False
            ),
            "random_baseline": task_config.params.get("random_baseline", False),
        }
        env_args = self._env_args(task_config, bddl_file)
        self.env = offscreen_env(**env_args)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        self.env.reset()

        self.last_native_obs = self.env.set_init_state(init_states[init_state_id])
        self._settle_after_init_state()
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        native_action = np.asarray(action.values, dtype=np.float32)
        obs, reward, done, info = self.env.step(native_action)
        self.step_id += 1
        self.last_native_obs = obs
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = self._jsonable_mapping(info)
        truncated = bool(self._max_steps and self.step_id >= self._max_steps and not self.last_done)

        return StepResult(
            observation=self._observation(),
            terminated=self.last_done,
            truncated=truncated,
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        image = self._extract_images(self.last_native_obs).get(camera.name)
        if image is not None:
            return image
        frame = self._as_hwc_uint8(
            self.env.sim.render(
                camera_name=camera.name,
                width=camera.width,
                height=camera.height,
                depth=False,
            )
        )
        return np.flipud(frame).copy()

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "reward": self.last_reward,
            "done": self.last_done,
            "is_success": self._check_success(),
            "benchmark_validity": dict(self._task_metadata),
        }
        state.update(self.last_info)
        eef_pos = self._obs_value("robot0_eef_pos")
        if eef_pos is not None:
            state["eef_pos"] = eef_pos
            state["ee_position"] = eef_pos
        return state

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _libero_modules(self) -> tuple[Any, Any]:
        self._ensure_libero_config()
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise ImportError(
                "LIBERO backend requires the pinned upstream LIBERO stack. Run "
                "`python3 scripts/manage_conda_env.py create libero` and "
                "`python3 scripts/manage_conda_env.py smoke libero`; see docs/libero_integration.md."
            ) from exc
        return benchmark, OffScreenRenderEnv

    def _ensure_libero_config(self) -> None:
        config_root = Path(os.environ.get("LIBERO_CONFIG_PATH", "/tmp/mllm_teleop_libero_config")).expanduser()
        os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
        config_file = config_root / "config.yaml"
        if config_file.exists():
            return

        env_root = os.environ.get("MLLM_TELEOP_LIBERO_BENCHMARK_ROOT")
        if env_root:
            benchmark_root = Path(env_root).expanduser().resolve()
        else:
            spec = importlib.util.find_spec("libero")
            benchmark_root = Path(tuple(spec.submodule_search_locations)[0]).resolve()
        config_root.mkdir(parents=True, exist_ok=True)
        with config_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "benchmark_root": str(benchmark_root),
                    "bddl_files": str(benchmark_root / "bddl_files"),
                    "init_states": str(benchmark_root / "init_files"),
                    "datasets": str(benchmark_root.parent / "datasets"),
                    "assets": str(benchmark_root / "assets"),
                },
                handle,
                sort_keys=True,
            )

    def _env_args(self, task_config: TaskConfig, bddl_file: str) -> dict[str, Any]:
        env_args = dict(self.config.env_kwargs)
        env_args.update(dict(self.config.params.get("env_args", {})))
        env_args["bddl_file_name"] = bddl_file
        env_args.setdefault("camera_names", self._camera_names())
        height, width = self._camera_dimensions()
        env_args.setdefault("camera_heights", height)
        env_args.setdefault("camera_widths", width)
        env_args.setdefault("use_camera_obs", True)
        policy_horizon = self.config.params.get(
            "max_episode_steps",
            self.config.params.get("horizon", task_config.max_steps),
        )
        required_native_horizon = policy_horizon + self.config.params.get("settling_steps", 5)
        # A shorter upstream horizon silently truncates the benchmark before its action budget.
        if "horizon" in env_args and env_args["horizon"] < required_native_horizon:
            raise ValueError(
                "LIBERO env_kwargs.horizon must include both the policy action "
                f"budget and settling steps; expected at least {required_native_horizon}"
            )
        env_args.setdefault("horizon", required_native_horizon)
        env_args.setdefault("control_freq", self.config.params.get("control_freq", 20))
        env_args.setdefault("hard_reset", self.config.params.get("hard_reset", True))
        if "render_gpu_device_id" in self.config.params:
            env_args.setdefault(
                "render_gpu_device_id", self.config.params["render_gpu_device_id"]
            )
        return env_args

    def _camera_names(self) -> list[str]:
        if self._camera_specs:
            return [camera.name for camera in self._camera_specs]
        raw_names = self.config.params.get("camera_names", ("agentview", "robot0_eye_in_hand"))
        return list(raw_names)

    def _camera_dimensions(self) -> tuple[int | list[int], int | list[int]]:
        if not self._camera_specs:
            return (
                self.config.params.get("camera_height", 128),
                self.config.params.get("camera_width", 128),
            )
        heights = [camera.height for camera in self._camera_specs]
        widths = [camera.width for camera in self._camera_specs]
        height: int | list[int] = heights[0] if len(set(heights)) == 1 else heights
        width: int | list[int] = widths[0] if len(set(widths)) == 1 else widths
        return height, width

    def _validate_language_instruction(self, task_config: TaskConfig, task: Any) -> None:
        configured_text = " ".join(task_config.params["language_instruction"].split())
        upstream_text = " ".join(task.language.split())
        # A mismatch otherwise runs one task while recording and prompting another.
        if configured_text != upstream_text:
            raise ValueError(
                "Configured LIBERO language instruction does not match the selected "
                f"upstream task: configured={configured_text!r}, upstream={upstream_text!r}"
            )

    def _settle_after_init_state(self) -> None:
        # reset() creates the environment immediately before settling it.
        assert self.env is not None
        settling_steps = self.config.params.get("settling_steps", 5)
        for _ in range(settling_steps):
            obs, _reward, done, info = self.env.step(
                np.zeros(7, dtype=np.float32)
            )
            self.last_native_obs = obs
            self.last_done = bool(done)
            self.last_info = self._jsonable_mapping(info)
        if settling_steps:
            self.last_done = self._check_success()
        self.last_reward = None

    def _select_init_state(
        self,
        task_config: TaskConfig,
        seed: int,
        init_state_count: int,
    ) -> tuple[int, str]:
        strategy = task_config.params.get(
            "init_state_strategy",
            self.config.params.get("init_state_strategy", "fixed"),
        )
        if {"fixed": False, "seed_modulo": True}[strategy]:
            return seed % init_state_count, strategy
        init_state_id = task_config.params.get(
            "init_state_id",
            self.config.params.get("init_state_id", 0),
        )
        return init_state_id, strategy

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images=self._extract_images(self.last_native_obs),
            state=self.get_state(),
            metadata={"simulator": "libero", "benchmark_validity": dict(self._task_metadata)},
        )

    def _extract_images(self, observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return {}
        images: dict[str, Any] = {}
        for key, value in observation.items():
            if key.endswith("_image"):
                camera_name = key[: -len("_image")]
                frame = self._as_hwc_uint8(value)
                if frame is not None:
                    images[camera_name] = np.flipud(frame).copy()
        return images

    def _obs_value(self, key: str) -> Any | None:
        if not isinstance(self.last_native_obs, dict) or key not in self.last_native_obs:
            return None
        return self._jsonable(self.last_native_obs[key])

    def _check_success(self) -> bool:
        if self.env is None:
            return False
        if hasattr(self.env, "check_success"):
            return bool(self.env.check_success())
        wrapped = getattr(self.env, "env", None)
        if wrapped is not None and hasattr(wrapped, "_check_success"):
            return bool(wrapped._check_success())
        return bool(self.last_info.get("success", False))

    def _as_hwc_uint8(self, value: Any) -> np.ndarray | None:
        array = self._to_numpy(value)
        if array is None:
            return None
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            return None
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _jsonable_mapping(self, value: Any) -> dict[str, Any]:
        converted = self._jsonable(value)
        return converted if isinstance(converted, dict) else {"info": converted}

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        array = self._to_numpy(value)
        if array is not None:
            if array.size == 1:
                return array.reshape(-1)[0].item()
            if array.size <= 32:
                return array.tolist()
            return {"array_shape": list(array.shape), "dtype": str(array.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.cpu().numpy()
        return None
