from __future__ import annotations

import os
from typing import Any

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import GymBoxAction, Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.gymnasium_envs.actions import action_to_native
from mllm_teleop_benchmark.sim.gymnasium_envs.registry import REGISTERED_REACHER_DELTA_EEF_ID, register_gymnasium_envs


class GymnasiumSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "gymnasium":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        if config.render_mode == "rgb_array":
            os.environ.setdefault("MUJOCO_GL", str(config.params.get("mujoco_gl", "egl")))
        import gymnasium as gym

        register_gymnasium_envs()
        self.config = config
        self.env_id = config.env_id or REGISTERED_REACHER_DELTA_EEF_ID
        kwargs = dict(config.env_kwargs)
        if config.render_mode is not None:
            kwargs["render_mode"] = config.render_mode
        if config.frame_skip is not None:
            kwargs["frame_skip"] = config.frame_skip
        if "max_episode_steps" in config.params:
            kwargs["max_episode_steps"] = int(config.params["max_episode_steps"])
        self.env = gym.make(self.env_id, **kwargs)
        self.step_id = 0
        self.last_native_obs: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_render: Any = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self.step_id = 0
        native_obs, info = self.env.reset(seed=seed)
        self.last_native_obs = native_obs
        self.last_info = dict(info)
        self.last_reward = None
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, GymBoxAction):
            raise TypeError(f"GymnasiumSimAdapter requires GymBoxAction, got {type(action).__name__}")
        native_action = action_to_native(action)
        native_obs, reward, terminated, truncated, info = self.env.step(native_action)
        self.step_id += 1
        self.last_native_obs = native_obs
        self.last_reward = float(reward)
        self.last_info = dict(info)
        return StepResult(
            observation=self._observation(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        frame = self.env.render()
        self.last_render = frame
        return frame

    def get_state(self) -> dict[str, Any]:
        info = dict(self.last_info)
        state = {
            "step_id": self.step_id,
            "env_id": self.env_id,
            "native_observation": self._jsonable_array(self.last_native_obs),
            "reward": self.last_reward,
        }
        state.update({key: self._jsonable_array(value) for key, value in info.items()})
        if "goal_distance" in state:
            state["distance_to_target"] = float(state["goal_distance"])
        return state

    def set_camera_configuration(self, camera_config: dict[str, object] | None) -> None:
        env = self.env.unwrapped
        if hasattr(env, "set_camera_configuration"):
            env.set_camera_configuration(camera_config)

    def close(self) -> None:
        self.env.close()

    def _observation(self) -> Observation:
        return Observation(step_id=self.step_id, images={}, state=self.get_state(), metadata={"simulator": "gymnasium", "env_id": self.env_id})

    def _jsonable_array(self, value: Any) -> Any:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return value
