from __future__ import annotations

import os
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import MetaworldAction, Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


DEFAULT_BENCHMARK_ID = "Meta-World/MT1"


class MetaworldSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "metaworld":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self.env_id = config.env_id or DEFAULT_BENCHMARK_ID
        self.env_name = str(config.params.get("env_name", config.env_kwargs.get("env_name", "")))
        if not self.env_name:
            raise ValueError("Meta-World simulator requires simulator.params.env_name or simulator.env_kwargs.env_name")
        self.env: Any | None = None
        self.step_id = 0
        self.last_native_obs: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_render: Any = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self._ensure_env(seed, task_config)
        self.step_id = 0
        reset_result = self.env.reset(seed=seed)
        native_obs, info = self._split_reset(reset_result)
        self.last_native_obs = native_obs
        self.last_info = self._jsonable_mapping(info)
        self.last_reward = None
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, MetaworldAction):
            raise TypeError(f"MetaworldSimAdapter requires MetaworldAction, got {type(action).__name__}")
        self._ensure_env(None, None)
        native_action, clip_metadata = self._native_action(action)
        step_result = self.env.step(native_action)
        native_obs, reward, terminated, truncated, info = self._split_step(step_result)
        self.step_id += 1
        self.last_native_obs = native_obs
        self.last_reward = float(reward)
        self.last_info = self._jsonable_mapping(info)
        if clip_metadata["clipped"]:
            self.last_info["action_clip"] = clip_metadata
        return StepResult(
            observation=self._observation(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        self._ensure_env(None, None)
        unwrapped = self.env.unwrapped
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "Meta-World rendering requires the MuJoCo Python package."
            ) from exc
        camera_id = mujoco.mj_name2id(
            unwrapped.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera.name,
        )
        if camera_id < 0:
            raise ValueError(f"Unknown Meta-World camera: {camera.name}")
        unwrapped.mujoco_renderer.camera_id = camera_id
        frame = self.env.render()
        self.last_render = self._first_array(frame)
        expected_shape = (camera.height, camera.width, 3)
        if self.last_render.shape != expected_shape:
            raise RuntimeError(
                "Meta-World render shape does not match the configured camera: "
                f"expected {expected_shape}, got {self.last_render.shape}"
            )
        return self.last_render

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "env_id": self.env_id,
            "env_name": self.env_name,
            "native_observation": self._jsonable(self.last_native_obs),
            "reward": self.last_reward,
        }
        state.update(self.last_info)
        ee_position = self._ee_position()
        if ee_position is not None:
            state["ee_position"] = ee_position
        if "obj_to_target" in state:
            state["distance_to_target"] = state["obj_to_target"]
        return state

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _ensure_env(self, seed: int | None, task_config: TaskConfig | None) -> None:
        if self.env is not None:
            return
        if self.config.render_mode == "rgb_array" or self.config.params.get("render_mode") == "rgb_array":
            os.environ.setdefault("MUJOCO_GL", str(self.config.params.get("mujoco_gl", "egl")))
        try:
            import gymnasium as gym
            import metaworld  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Meta-World backend requires the optional metaworld dependency. "
                "Create the metaworld uv environment before using simulator.type=metaworld."
            ) from exc

        kwargs = self._make_kwargs(seed, task_config)
        self.env = gym.make(self.env_id, **kwargs)
        expected_horizon = self.config.params.get("max_episode_steps")
        actual_horizon = getattr(self.env, "_max_episode_steps", None)
        if (
            expected_horizon is not None
            and actual_horizon is not None
            and int(actual_horizon) < int(expected_horizon)
        ):
            self.close()
            raise RuntimeError(
                "Meta-World environment horizon is smaller than the required "
                f"simulator execution budget: {actual_horizon} < {expected_horizon}"
            )

    def _make_kwargs(self, seed: int | None, task_config: TaskConfig | None) -> dict[str, Any]:
        kwargs = dict(self.config.env_kwargs)
        kwargs.setdefault("env_name", self.env_name)
        if self.config.render_mode is not None:
            kwargs.setdefault("render_mode", self.config.render_mode)
        if "render_mode" in self.config.params:
            kwargs.setdefault("render_mode", self.config.params["render_mode"])
        if "camera_name" in self.config.params:
            kwargs.setdefault("camera_name", self.config.params["camera_name"])
        if "camera_id" in self.config.params:
            kwargs.setdefault("camera_id", self.config.params["camera_id"])
        if "width" in self.config.params:
            kwargs.setdefault("width", int(self.config.params["width"]))
        if "height" in self.config.params:
            kwargs.setdefault("height", int(self.config.params["height"]))
        if "max_episode_steps" in self.config.params:
            kwargs.setdefault("max_episode_steps", int(self.config.params["max_episode_steps"]))
        elif task_config is not None:
            kwargs.setdefault("max_episode_steps", int(task_config.max_steps))
        if seed is not None:
            kwargs.setdefault("seed", int(seed))
        return kwargs

    def _native_action(self, action: MetaworldAction) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(action.values, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Meta-World action contains non-finite values")
        action_space = getattr(self.env, "action_space", None)
        metadata: dict[str, Any] = {"clipped": False}
        if action_space is None or not hasattr(action_space, "shape"):
            return values, metadata
        target_shape = tuple(int(value) for value in action_space.shape)
        if values.shape != target_shape:
            raise ValueError(f"Meta-World action shape mismatch: controller produced {values.shape}, env expects {target_shape}")
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            before = values.copy()
            values = np.clip(values, action_space.low, action_space.high).astype(np.float32)
            clipped = bool(np.any(before != values))
            metadata = {
                "clipped": clipped,
                "before": before.tolist() if clipped else None,
                "after": values.tolist() if clipped else None,
            }
        return values, metadata

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={"simulator": "metaworld", "env_id": self.env_id, "env_name": self.env_name},
        )

    def _split_reset(self, value: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(value, tuple) and len(value) == 2:
            return value[0], dict(value[1])
        return value, {}

    def _split_step(self, value: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if isinstance(value, tuple) and len(value) == 5:
            obs, reward, terminated, truncated, info = value
            return obs, float(reward), bool(terminated), bool(truncated), dict(info)
        if isinstance(value, tuple) and len(value) == 4:
            obs, reward, done, info = value
            return obs, float(reward), bool(done), False, dict(info)
        raise ValueError(f"Unexpected Meta-World step result: {value!r}")

    def _jsonable_mapping(self, value: Any) -> dict[str, Any]:
        converted = self._jsonable(value)
        return converted if isinstance(converted, dict) else {"info": converted}

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            if value.size <= 32:
                return value.tolist()
            return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _first_array(self, value: Any) -> Any:
        if isinstance(value, list | tuple) and value:
            value = value[0]
        if isinstance(value, np.ndarray) and value.ndim >= 4 and value.shape[0] == 1:
            return value[0]
        return value

    def _ee_position(self) -> list[float] | None:
        if not isinstance(self.last_native_obs, np.ndarray) or self.last_native_obs.size < 3:
            return None
        return [float(value) for value in self.last_native_obs.reshape(-1)[:3]]
