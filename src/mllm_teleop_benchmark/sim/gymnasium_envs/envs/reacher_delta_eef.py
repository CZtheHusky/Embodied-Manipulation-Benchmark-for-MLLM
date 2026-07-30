from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from mllm_teleop_benchmark.sim.gymnasium_envs.camera import MujocoCameraConfigurator


class PlanarReacherKinematics:
    def __init__(self, env: MujocoEnv, link1_len: float, link2_len: float) -> None:
        self.env = env
        self.r_min = max(0.0, abs(float(link1_len) - float(link2_len)) + 1e-4)
        self.r_max = max(1e-6, float(link1_len) + float(link2_len) - 1e-4)

    def clamp_world_xy(self, target_xy: np.ndarray) -> tuple[np.ndarray, bool]:
        xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        xy = np.clip(xy, -0.27, 0.27)
        radius = float(np.linalg.norm(xy))
        if radius < 1e-12:
            return np.array([self.r_min, 0.0], dtype=np.float64), True
        if radius < self.r_min:
            return xy * (self.r_min / radius), True
        if radius > self.r_max:
            return xy * (self.r_max / radius), True
        return xy, False

    def solve_xy(self, target_xy: np.ndarray, iters: int = 15, step_scale: float = 0.2) -> np.ndarray:
        qpos = self.env.data.qpos.copy()
        jacp = np.zeros((3, self.env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.env.model.nv), dtype=np.float64)
        for _ in range(max(1, int(iters))):
            self.env.data.qpos[:] = qpos
            mujoco.mj_forward(self.env.model, self.env.data)
            tip_xy = self.env.get_body_com("fingertip")[:2]
            err = np.asarray(target_xy, dtype=np.float64) - tip_xy
            if float(np.linalg.norm(err)) < 1e-4:
                break
            mujoco.mj_jacBody(self.env.model, self.env.data, jacp, jacr, self.env.model.body("fingertip").id)
            jxy = jacp[:2, :2]
            try:
                dq = step_scale * (np.linalg.pinv(jxy) @ err)
            except np.linalg.LinAlgError:
                dq = np.zeros(2, dtype=np.float64)
            qpos[:2] += dq
            qpos[0] = float(np.clip(qpos[0], -3.14, 3.14))
            qpos[1] = float(np.clip(qpos[1], -3.0, 3.0))
        self.env.data.qpos[:] = qpos
        mujoco.mj_forward(self.env.model, self.env.data)
        return qpos[:2].copy()


class GoalTermination:
    def __init__(self, threshold: float, hold_steps: int) -> None:
        self.threshold = float(threshold)
        self.hold_steps = int(hold_steps)
        self.counter = 0

    def reset(self) -> None:
        self.counter = 0

    def update(self, distance: float) -> tuple[bool, bool]:
        if float(distance) <= self.threshold:
            self.counter += 1
        else:
            self.counter = 0
        reached = self.counter >= self.hold_steps if self.hold_steps >= 0 else False
        terminated = reached if self.hold_steps >= 0 else False
        return reached, terminated


class ReacherDeltaEEFEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        camera_configuration: dict[str, Any] | None = None,
        eef_step_scale: float = 0.01,
        link1_len: float = 0.1,
        link2_len: float = 0.11,
        goal_distance_threshold: float = 0.015,
        goal_hold_steps: int = 5,
        random_goal: bool = True,
        frame_skip: int = 1,
        **kwargs: Any,
    ) -> None:
        self.eef_step_scale = float(eef_step_scale)
        self.random_goal = bool(random_goal)
        self.fixed_goal: np.ndarray | None = None
        self.target_eef_xy: np.ndarray | None = None
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_target_clamped = False
        self.termination = GoalTermination(goal_distance_threshold, goal_hold_steps)
        self.camera_configurator = MujocoCameraConfigurator()
        self.camera_configuration = self.camera_configurator.prepare(camera_configuration)

        xml_path = Path(__file__).with_name("assets") / "reacher_pos.xml"
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        MujocoEnv.__init__(self, str(xml_path), frame_skip=frame_skip, observation_space=observation_space, render_mode=render_mode)
        utils.EzPickle.__init__(
            self,
            render_mode=render_mode,
            camera_configuration=camera_configuration,
            eef_step_scale=eef_step_scale,
            link1_len=link1_len,
            link2_len=link2_len,
            goal_distance_threshold=goal_distance_threshold,
            goal_hold_steps=goal_hold_steps,
            random_goal=random_goal,
            frame_skip=frame_skip,
            **kwargs,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.kinematics = PlanarReacherKinematics(self, link1_len, link2_len)
        self._apply_camera_configuration()

    def set_camera_configuration(self, camera_configuration: dict[str, Any] | None) -> None:
        self.camera_configuration = self.camera_configurator.prepare(camera_configuration)
        self._apply_camera_configuration()

    def get_goal_position(self) -> np.ndarray:
        return self.get_body_com("target")[:2].astype(np.float64).copy()

    def set_goal_position_and_reset(self, goal_xy: Any):
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        self.random_goal = False
        self.fixed_goal = goal.copy()
        self.init_qpos[-2:] = self.fixed_goal
        return self.reset()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._apply_camera_configuration()
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info.update(self._info())
        return obs, info

    def reset_model(self):
        self.termination.reset()
        noise = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qpos = self.init_qpos + noise
        if self.random_goal or self.fixed_goal is None:
            while True:
                goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
                if float(np.linalg.norm(goal)) < 0.2:
                    break
            if not self.random_goal:
                self.fixed_goal = np.asarray(goal, dtype=np.float64).copy()
        else:
            goal = self.fixed_goal
        qpos[-2:] = np.asarray(goal, dtype=np.float64)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)
        self.target_eef_xy = self.get_body_com("fingertip")[:2].astype(np.float64)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.last_target_clamped = False
        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos[:2].copy()
        tip_xy = self.get_body_com("fingertip")[:2]
        target_xy = self.get_body_com("target")[:2]
        return np.concatenate([qpos, tip_xy, target_xy]).astype(np.float32)

    def step(self, action):
        self._apply_camera_configuration()
        array = np.asarray(action, dtype=np.float32).reshape(-1)
        if array.size != 2:
            raise ValueError(f"Reacher delta EEF action must be 2D, got {array.shape}")
        array = np.clip(array, -1.0, 1.0)
        delta_xy = array.astype(np.float64) * self.eef_step_scale
        if self.target_eef_xy is None:
            self.target_eef_xy = self.get_body_com("fingertip")[:2].astype(np.float64)
        target_xy, clamped = self.kinematics.clamp_world_xy(self.target_eef_xy + delta_xy)
        ctrl = self.kinematics.solve_xy(target_xy)
        self.target_eef_xy = target_xy
        self.last_target_clamped = clamped
        self.last_action = array.copy()
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        distance = self._distance()
        reached, terminated = self.termination.update(distance)
        reward = 0.0 if reached else -distance
        info = self._info()
        info["goal_termination"] = reached
        return obs, reward, terminated, False, info

    def _distance(self) -> float:
        return float(np.linalg.norm(self.get_body_com("fingertip") - self.get_body_com("target")))

    def _info(self) -> dict[str, Any]:
        return {
            "goal_distance": self._distance(),
            "goal_hold_counter": self.termination.counter,
            "fingertip_position": self.get_body_com("fingertip").astype(np.float32),
            "target_position": self.get_body_com("target").astype(np.float32),
            "ee_target_xy": np.asarray(self.target_eef_xy if self.target_eef_xy is not None else [0.0, 0.0], dtype=np.float32),
            "ee_target_reachability_clamped": bool(self.last_target_clamped),
            "last_action": self.last_action.astype(np.float32),
        }

    def _apply_camera_configuration(self) -> None:
        if hasattr(self, "mujoco_renderer"):
            self.camera_configurator.apply(self.mujoco_renderer, self.camera_configuration)
