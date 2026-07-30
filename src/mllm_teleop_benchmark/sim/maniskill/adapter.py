from __future__ import annotations

from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import ManiSkillAction, Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class ManiSkillSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "maniskill":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self.env_id = config.env_id
        if self.env_id is None:
            raise ValueError("ManiSkill simulator requires simulator.env_id")
        self.env: Any | None = None
        self.step_id = 0
        self.last_native_obs: Any = None
        self.last_info: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_render: Any = None
        self._camera_config: dict[str, Any] | None = None

    def set_camera_configuration(self, camera_config: dict[str, Any] | None) -> None:
        self._camera_config = dict(camera_config) if camera_config is not None else None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self._ensure_env(task_config)
        self.step_id = 0
        native_obs, info = self.env.reset(seed=seed)
        self.last_native_obs = native_obs
        self.last_info = self._jsonable_mapping(info)
        self.last_reward = None
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, ManiSkillAction):
            raise TypeError(f"ManiSkillSimAdapter requires ManiSkillAction, got {type(action).__name__}")
        self._ensure_env(None)
        native_action, clip_metadata = self._native_action(action)
        native_obs, reward, terminated, truncated, info = self.env.step(native_action)
        self.step_id += 1
        self.last_native_obs = native_obs
        self.last_reward = self._float_or_none(reward)
        self.last_info = self._jsonable_mapping(info)
        if clip_metadata["clipped"]:
            self.last_info["action_clip"] = clip_metadata
        return StepResult(
            observation=self._observation(),
            terminated=self._bool_value(terminated),
            truncated=self._bool_value(truncated),
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        self._ensure_env(None)
        if hasattr(self.env, "render"):
            frame = self.env.render()
            self.last_render = self._first_array(frame)
            return self.last_render
        raise RuntimeError("ManiSkill environment does not expose render()")

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "env_id": self.env_id,
            "native_observation": self._jsonable(self.last_native_obs),
            "reward": self.last_reward,
        }
        state.update(self.last_info)
        if "success" in state:
            state["success"] = self._bool_value(state["success"])
        tcp_position = self._tcp_position()
        if tcp_position is not None:
            state["ee_position"] = tcp_position
            state["tcp_position"] = tcp_position
        return state

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _ensure_env(self, task_config: TaskConfig | None) -> None:
        if self.env is not None:
            return
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ManiSkill backend requires the optional mani-skill dependency. "
                "Install it in the benchmark conda env before using simulator.type=maniskill."
            ) from exc

        kwargs = dict(self.config.env_kwargs)
        kwargs.setdefault("obs_mode", self.config.params.get("obs_mode", "rgb"))
        kwargs.setdefault("render_mode", self.config.render_mode or self.config.params.get("render_mode", "rgb_array"))
        if "control_mode" in self.config.params:
            kwargs.setdefault("control_mode", self.config.params["control_mode"])
        if task_config is not None and "max_episode_steps" not in kwargs:
            kwargs["max_episode_steps"] = int(self.config.params.get("max_episode_steps", task_config.max_steps))
        if self._camera_config:
            kwargs.update(self._camera_config)
        kwargs = self._convert_maniskill_kwargs(kwargs)

        env = gym.make(self.env_id, **kwargs)
        self.env = self._wrap_cpu(env)
        expected_horizon = self.config.params.get("max_episode_steps")
        actual_horizon = getattr(env, "_max_episode_steps", None)
        if (
            expected_horizon is not None
            and actual_horizon is not None
            and int(actual_horizon) < int(expected_horizon)
        ):
            self.close()
            raise RuntimeError(
                "ManiSkill environment horizon is smaller than the required "
                f"simulator execution budget: {actual_horizon} < {expected_horizon}"
            )

    def _wrap_cpu(self, env: Any) -> Any:
        if "cpu_gym_wrapper" not in self.config.wrappers:
            return env
        try:
            from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        except ImportError:
            return env
        return CPUGymWrapper(env)

    def _convert_maniskill_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "sensor_configs" not in kwargs:
            return kwargs
        try:
            from sapien import Pose
        except ImportError:
            return kwargs
        converted = dict(kwargs)
        sensor_configs = {}
        for name, config in dict(kwargs["sensor_configs"]).items():
            entry = dict(config)
            pose = entry.get("pose")
            if isinstance(pose, dict) and "p" in pose and "q" in pose:
                entry["pose"] = Pose(p=pose["p"], q=pose["q"])
            sensor_configs[name] = entry
        converted["sensor_configs"] = sensor_configs
        return converted

    def _native_action(self, action: ManiSkillAction) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(action.values, dtype=np.float32)
        action_space = getattr(self.env, "action_space", None)
        metadata: dict[str, Any] = {"clipped": False}
        if action_space is None or not hasattr(action_space, "shape"):
            return values, metadata
        target_shape = tuple(int(v) for v in action_space.shape)
        if values.shape != target_shape:
            if len(target_shape) == 2 and target_shape[0] == 1 and values.shape == target_shape[1:]:
                values = values.reshape(target_shape)
            else:
                raise ValueError(f"ManiSkill action shape mismatch: controller produced {values.shape}, env expects {target_shape}")
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            before = values.copy()
            values = np.clip(values, action_space.low, action_space.high)
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
            metadata={"simulator": "maniskill", "env_id": self.env_id},
        )

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
        return value

    def _first_array(self, value: Any) -> Any:
        array = self._to_numpy(value)
        if array is None:
            return value
        if array.ndim >= 4 and array.shape[0] == 1:
            array = array[0]
        return array

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.cpu().numpy()
        if isinstance(value, np.generic):
            return np.asarray(value)
        return None

    def _float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        array = self._to_numpy(value)
        if array is not None:
            return float(array.reshape(-1)[0])
        return float(value)

    def _bool_value(self, value: Any) -> bool:
        array = self._to_numpy(value)
        if array is not None:
            return bool(array.reshape(-1)[0])
        return bool(value)

    def _tcp_position(self) -> list[float] | None:
        if self.env is None:
            return None
        try:
            unwrapped = getattr(self.env, "unwrapped", self.env)
            agent = getattr(unwrapped, "agent", None)
            tcp = getattr(agent, "tcp", None)
            pose = getattr(tcp, "pose", None)
            position = getattr(pose, "p", None)
            array = self._to_numpy(position)
            if array is None:
                return None
            return [float(v) for v in array.reshape(-1)[:3]]
        except Exception:
            return None
