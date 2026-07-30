from __future__ import annotations

from collections.abc import Mapping
import random
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, RoboCasaAction, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


DEFAULT_ENV_ID = "robocasa/PickPlaceCounterToCabinet"


class RoboCasaSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "robocasa":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self._env = None
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info: dict[str, Any] = {}
        self._last_obs_state: dict[str, Any] = {}
        self._last_obs_images: dict[str, Any] = {}
        self._step_id = 0

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        gym = self._robocasa_gym_module()
        random.seed(seed)
        np.random.seed(int(seed) % (2**32))

        self.close()
        env_id = self.config.env_id or str(task_config.params.get("env_id", DEFAULT_ENV_ID))
        env_kwargs = self._env_kwargs(task_config, seed)
        self._env = gym.make(env_id, **env_kwargs)
        expected_horizon = self.config.params.get("max_episode_steps")
        actual_horizon = getattr(self._env, "_max_episode_steps", None)
        if (
            expected_horizon is not None
            and actual_horizon is not None
            and int(actual_horizon) < int(expected_horizon)
        ):
            self.close()
            raise RuntimeError(
                "RoboCasa horizon is smaller than the required simulator "
                f"execution budget: {actual_horizon} < {expected_horizon}"
            )
        obs, info = self._reset_env(seed)

        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = dict(info)
        self._last_info["success"] = self._native_success()
        self._step_id = 0
        return self._format_observation(obs)

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, RoboCasaAction):
            raise TypeError(f"RoboCasaSimAdapter requires RoboCasaAction, got {type(action).__name__}")
        obs, reward, terminated, truncated, info = self._env.step(_coerce_action_dict(action.values))
        self._step_id += 1
        self._last_reward = float(reward)
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._last_info = dict(info)

        return StepResult(
            observation=self._format_observation(obs),
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            info=self._last_info,
        )

    def render(self, camera: CameraSpec) -> Any:
        if camera.name in self._last_obs_images:
            return self._last_obs_images[camera.name]
        video_key = f"video.{camera.name}"
        if video_key in self._last_obs_images:
            return self._last_obs_images[video_key]
        image_key = f"{camera.name}_image"
        if image_key in self._last_obs_images:
            return self._last_obs_images[image_key]

        rendered = self._env.render()
        if rendered is None:
            raise RuntimeError(f"RoboCasa render returned no image for camera {camera.name!r}")
        return rendered

    def get_state(self) -> dict[str, Any]:
        return {
            "step_id": self._step_id,
            "reward": self._last_reward,
            "terminated": self._last_terminated,
            "truncated": self._last_truncated,
            "is_success": self._success(),
            "info": self._last_info,
            "observation": self._last_obs_state,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _env_kwargs(self, task_config: TaskConfig, seed: int) -> dict[str, Any]:
        kwargs = dict(self.config.env_kwargs)
        task_env_kwargs = task_config.params.get("env_kwargs", {})
        if not isinstance(task_env_kwargs, dict):
            raise ValueError("task.params.env_kwargs must be a mapping when provided")
        kwargs.update(task_env_kwargs)
        if "split" in task_config.params and "split" not in kwargs:
            kwargs["split"] = task_config.params["split"]
        kwargs.setdefault("split", "pretrain")
        kwargs.setdefault("seed", seed)
        kwargs.setdefault(
            "max_episode_steps",
            int(
                self.config.params.get(
                    "max_episode_steps",
                    task_config.max_steps,
                )
            ),
        )
        return kwargs

    def _reset_env(self, seed: int) -> tuple[Any, dict[str, Any]]:
        try:
            result = self._env.reset(seed=seed)
        except TypeError:
            result = self._env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs, dict(info)
        return result, {}

    def _format_observation(self, obs: Any) -> Observation:
        obs_dict = dict(obs) if isinstance(obs, Mapping) else {"observation": obs}
        images: dict[str, Any] = {}
        state: dict[str, Any] = {}
        for key, value in obs_dict.items():
            if _is_image(value):
                images[str(key)] = value
                normalized = _normalized_image_name(str(key))
                images.setdefault(normalized, value)
            else:
                state[str(key)] = value

        self._last_obs_images = images
        self._last_obs_state = state
        return Observation(
            step_id=self._step_id,
            images=images,
            state=state,
            metadata={
                "env_id": self.config.env_id or DEFAULT_ENV_ID,
                "success": self._success(),
            },
        )

    def _success(self) -> bool:
        if "success" in self._last_info:
            return bool(self._last_info["success"])
        if self._env is not None:
            return self._native_success()
        return False

    def _native_success(self) -> bool:
        pending = [getattr(self._env, "unwrapped", None), self._env]
        visited: set[int] = set()
        while pending:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            check_success = getattr(candidate, "_check_success", None)
            if callable(check_success):
                return bool(check_success())
            pending.extend(
                (
                    getattr(candidate, "unwrapped", None),
                    getattr(candidate, "env", None),
                )
            )
        raise RuntimeError(
            "RoboCasa environment does not expose the native _check_success() oracle"
        )

    def _robocasa_gym_module(self) -> Any:
        try:
            import gymnasium as gym
            import robocasa  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RoboCasa backend requires the isolated robocasa uv environment. "
                "Run `python3 scripts/manage_uv_env.py create robocasa`."
            ) from exc
        return gym


def _coerce_action_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): np.asarray(value, dtype=np.float32) for key, value in values.items()}


def _is_image(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in {3, 4}


def _normalized_image_name(key: str) -> str:
    if key.startswith("video."):
        return key[len("video.") :]
    if key.endswith("_image"):
        return key[: -len("_image")]
    return key
