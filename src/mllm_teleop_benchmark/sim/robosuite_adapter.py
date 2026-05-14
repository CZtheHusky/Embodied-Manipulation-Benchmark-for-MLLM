from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
from robosuite.utils.mjmod import CameraModder

from mllm_teleop_benchmark.config.schema import CameraSpec, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class RobosuiteSimAdapter(SimAdapter):
    def __init__(self) -> None:
        self._env = None
        self._modder = None
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info: dict[str, Any] = {}
        self._step_id = 0

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        # robosuite uses seed through numpy randomization or env-specific seed
        # For simplicity and reproducibility, we attempt to pass it if supported
        
        # Determine robot and env from config params if available, else defaults
        env_name = task_config.params.get("env_name", "Lift")
        robots = task_config.params.get("robots", ["Panda"])
        
        # Extract camera names and resolutions from task_config if available
        # In our benchmark, they might be in a separate 'camera' section, 
        # but here we are in SimAdapter.reset(task_config, ...).
        # We'll use defaults or try to find them in task_config.params.
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
            ignore_done=False,
            hard_reset=False,
        )
        
        obs_dict = self._env.reset()
        self._modder = CameraModder(self._env.sim)
        self._step_id = 0
        
        # Apply camera randomization if configured in task_config.params
        if "camera_randomization" in task_config.params:
            rand_cfg = task_config.params["camera_randomization"]
            for cam_name, params in rand_cfg.items():
                if "pos" in params:
                    self._modder.set_pos(cam_name, np.array(params["pos"]))
                if "quat" in params:
                    self._modder.set_quat(cam_name, np.array(params["quat"]))
                if "fovy" in params:
                    self._modder.set_fovy(cam_name, float(params["fovy"]))
            self._env.sim.forward()
            # Re-fetch observation after camera move
            obs_dict = self._env._get_observations()

        return self._format_observation(obs_dict)

    def step(self, action: SimulatorAction) -> StepResult:
        # action is expected to be a numpy array for robosuite
        obs_dict, reward, done, info = self._env.step(action)
        self._step_id += 1
        self._last_reward = float(reward)
        # robosuite's 'done' is equivalent to terminated for most tasks
        # gymnasium 1.0 style expects separated terminated/truncated
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
        # robosuite returns images in observations when use_camera_obs=True
        # but we can fetch them via render() or _get_observations()
        return self._env.sim.render(
            camera_name=camera.name,
            width=camera.width,
            height=camera.height,
            depth=False,
        )

    def get_state(self) -> dict[str, Any]:
        obs = self._env._get_observations()
        return {
            "robot_proprio": obs.get("robot0_proprio-state"),
            "eef_pos": obs.get("robot0_eef_pos"),
            "eef_quat": obs.get("robot0_eef_quat"),
            "is_success": self._env._check_success(),
            "reward": self._last_reward,
            "info": self._last_info,
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
        )
