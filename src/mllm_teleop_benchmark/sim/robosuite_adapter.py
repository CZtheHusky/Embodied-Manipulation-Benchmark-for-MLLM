from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, RobosuiteAction, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class RobosuiteSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "robosuite":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self._env = None
        self._modder = None
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info: dict[str, Any] = {}
        self._step_id = 0
        self._camera_randomization_sample: dict[str, dict[str, Any]] = {}

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        suite, camera_modder_cls = self._robosuite_modules()
        random.seed(seed)
        np.random.seed(int(seed) % (2**32))
        self.close()

        env_name = task_config.params.get("env_name", "Lift")
        robots = task_config.params.get("robots", ["Panda"])
        camera_names = task_config.params.get("camera_names", ["agentview", "robot0_eye_in_hand"])
        camera_width = task_config.params.get("camera_width", 320)
        camera_height = task_config.params.get("camera_height", 240)

        self._env = suite.make(
            env_name=env_name,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_widths=camera_width,
            camera_heights=camera_height,
            controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
            horizon=int(
                self.config.params.get(
                    "max_episode_steps",
                    task_config.max_steps,
                )
            ),
            ignore_done=True,
            hard_reset=False,
        )

        if hasattr(self._env, "seed"):
            try:
                self._env.seed(seed)
            except TypeError:
                self._env.seed()
        try:
            obs_dict = self._env.reset(seed=seed)
        except TypeError:
            obs_dict = self._env.reset()
        self._modder = camera_modder_cls(self._env.sim)
        self._step_id = 0
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = {}
        self._camera_randomization_sample = {}

        if "camera_randomization" in task_config.params:
            sampled_cfg = sample_camera_randomization(task_config.params["camera_randomization"], seed)
            self._camera_randomization_sample = sampled_cfg
            for cam_name, params in sampled_cfg.items():
                if "pos" in params:
                    self._modder.set_pos(cam_name, np.array(params["pos"]))
                if "quat" in params:
                    self._modder.set_quat(cam_name, np.array(params["quat"]))
                if "fovy" in params:
                    self._modder.set_fovy(cam_name, float(params["fovy"]))
            self._env.sim.forward()
            obs_dict = self._env._get_observations()

        return self._format_observation(obs_dict)

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, RobosuiteAction):
            raise TypeError(f"RobosuiteSimAdapter requires RobosuiteAction, got {type(action).__name__}")
        obs_dict, reward, done, info = self._env.step(np.asarray(action.values, dtype=np.float32))
        self._step_id += 1
        self._last_reward = float(reward)
        self._last_terminated = bool(done)
        self._last_truncated = False
        self._last_info = info

        return StepResult(
            observation=self._format_observation(obs_dict),
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            info=info,
        )

    def render(self, camera: CameraSpec) -> Any:
        return np.flipud(
            self._env.sim.render(
                camera_name=camera.name,
                width=camera.width,
                height=camera.height,
                depth=False,
            )
        )

    def get_state(self) -> dict[str, Any]:
        obs = self._env._get_observations()
        return {
            "step_id": self._step_id,
            "robot_proprio": obs.get("robot0_proprio-state"),
            "eef_pos": obs.get("robot0_eef_pos"),
            "eef_quat": obs.get("robot0_eef_quat"),
            "is_success": bool(self._env._check_success()),
            "reward": self._last_reward,
            "info": self._last_info,
            "camera_randomization": self._camera_randomization_sample,
        }

    def _format_observation(self, obs_dict: dict[str, Any]) -> Observation:
        # Extract images from obs_dict (robosuite names them camera_name + "_image")
        # and flip them as MuJoCo renders upside down by default
        images = {}
        for key, value in obs_dict.items():
            if key.endswith("_image"):
                cam_name = key[: -len("_image")]
                images[cam_name] = np.flipud(value)

        return Observation(
            step_id=self._step_id,
            images=images,
            state=obs_dict,
            metadata={"camera_randomization": self._camera_randomization_sample},
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _robosuite_modules(self) -> tuple[Any, Any]:
        try:
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
            import robosuite as suite
            from robosuite.utils.mjmod import CameraModder
        except ImportError as exc:
            raise ImportError(
                "Robosuite backend requires the mllm-teleop-robosuite conda environment "
                "or the optional robosuite dependency."
            ) from exc
        return suite, CameraModder


def sample_camera_randomization(config: object, seed: int) -> dict[str, dict[str, Any]]:
    if not isinstance(config, dict):
        raise ValueError("robosuite camera_randomization must be a mapping")
    rng = random.Random(seed)
    sampled: dict[str, dict[str, Any]] = {}
    for cam_name, raw_params in config.items():
        if not isinstance(raw_params, dict):
            raise ValueError(f"camera_randomization.{cam_name} must be a mapping")
        params: dict[str, Any] = {}
        if "pos" in raw_params:
            params["pos"] = _fixed_vector(raw_params["pos"], "pos", 3)
        if "pos_range" in raw_params:
            params["pos"] = [_sample_range(rng, value, "pos_range") for value in _range_vector(raw_params["pos_range"], "pos_range", 3)]
        if "quat" in raw_params:
            params["quat"] = _fixed_vector(raw_params["quat"], "quat", 4)
        if "fovy" in raw_params:
            params["fovy"] = float(raw_params["fovy"])
        if "fovy_range" in raw_params:
            params["fovy"] = _sample_range(rng, raw_params["fovy_range"], "fovy_range")
        sampled[str(cam_name)] = params
    return sampled


def _fixed_vector(value: object, name: str, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numeric values")
    return [float(item) for item in value]


def _range_vector(value: object, name: str, length: int) -> list[object]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{name} must contain {length} ranges")
    return list(value)


def _sample_range(rng: random.Random, value: object, name: str) -> float:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{name} entries must be [low, high]")
    return rng.uniform(float(value[0]), float(value[1]))
