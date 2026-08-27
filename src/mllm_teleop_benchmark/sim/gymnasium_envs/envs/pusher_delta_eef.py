from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from mllm_teleop_benchmark.sim.gymnasium_envs.camera import MujocoCameraConfigurator
from mllm_teleop_benchmark.sim.gymnasium_envs.envs.reacher_delta_eef import GoalTermination


ARM_JOINT_COUNT = 7
TIP_BODY = "tips_arm"
OBJECT_BODY = "object"
GOAL_BODY = "goal"


class PusherDeltaEEFIK:
    def __init__(
        self,
        env: MujocoEnv,
        *,
        damping: float = 0.05,
        step_scale: float = 0.75,
        max_joint_delta: float = 0.12,
        iters: int = 20,
    ) -> None:
        self.env = env
        self.damping = float(damping)
        self.step_scale = float(step_scale)
        self.max_joint_delta = float(max_joint_delta)
        self.iters = int(iters)

    def solve_xyz(self, target_xyz: np.ndarray) -> np.ndarray:
        saved_qpos = self.env.data.qpos.copy()
        saved_qvel = self.env.data.qvel.copy()
        qpos = saved_qpos.copy()
        jacp = np.zeros((3, self.env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.env.model.nv), dtype=np.float64)
        body_id = self.env.model.body(TIP_BODY).id

        for _ in range(max(1, self.iters)):
            self.env.data.qpos[:] = qpos
            self.env.data.qvel[:] = 0.0
            mujoco.mj_forward(self.env.model, self.env.data)
            err = np.asarray(target_xyz, dtype=np.float64).reshape(3) - self.env.get_body_com(TIP_BODY)
            if float(np.linalg.norm(err)) < 1e-4:
                break
            mujoco.mj_jacBody(self.env.model, self.env.data, jacp, jacr, body_id)
            jacobian = jacp[:, :ARM_JOINT_COUNT]
            lhs = jacobian @ jacobian.T + (self.damping**2) * np.eye(3, dtype=np.float64)
            try:
                dq = jacobian.T @ np.linalg.solve(lhs, err)
            except np.linalg.LinAlgError:
                dq = np.zeros(ARM_JOINT_COUNT, dtype=np.float64)
            dq *= self.step_scale
            max_abs = float(np.max(np.abs(dq))) if dq.size else 0.0
            if max_abs > self.max_joint_delta > 0:
                dq *= self.max_joint_delta / max_abs
            qpos[:ARM_JOINT_COUNT] += dq
            qpos[:ARM_JOINT_COUNT] = self._clip_arm_qpos(qpos[:ARM_JOINT_COUNT])

        self.env.data.qpos[:] = saved_qpos
        self.env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        return qpos[:ARM_JOINT_COUNT].copy()

    def _clip_arm_qpos(self, qpos: np.ndarray) -> np.ndarray:
        clipped = np.asarray(qpos, dtype=np.float64).copy()
        for idx in range(ARM_JOINT_COUNT):
            joint_range = self.env.model.jnt_range[idx]
            if self.env.model.jnt_limited[idx]:
                clipped[idx] = float(np.clip(clipped[idx], joint_range[0], joint_range[1]))
        return clipped


class PusherDeltaEEFEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"], "render_fps": 20}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        camera_configuration: dict[str, Any] | None = None,
        eef_step_scale: float = 0.01,
        workspace_low: tuple[float, float, float] = (0.1, -0.65, -0.25),
        workspace_high: tuple[float, float, float] = (0.95, 0.35, 0.25),
        goal_distance_threshold: float = 0.08,
        goal_hold_steps: int = 3,
        ik_damping: float = 0.05,
        ik_step_scale: float = 0.75,
        ik_max_joint_delta: float = 0.12,
        ik_iters: int = 20,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1.0,
        reward_control_weight: float = 0.02,
        frame_skip: int = 5,
        width: int = 512,
        height: int = 512,
        **kwargs: Any,
    ) -> None:
        self.eef_step_scale = float(eef_step_scale)
        self.workspace_low = np.asarray(workspace_low, dtype=np.float64).reshape(3)
        self.workspace_high = np.asarray(workspace_high, dtype=np.float64).reshape(3)
        self.reward_near_weight = float(reward_near_weight)
        self.reward_dist_weight = float(reward_dist_weight)
        self.reward_control_weight = float(reward_control_weight)
        self.target_eef_xyz: np.ndarray | None = None
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_ctrl = np.zeros(ARM_JOINT_COUNT, dtype=np.float32)
        self.last_target_clamped = False
        self.termination = GoalTermination(goal_distance_threshold, goal_hold_steps)
        self.camera_configurator = MujocoCameraConfigurator()
        self.camera_configuration = self.camera_configurator.prepare(camera_configuration)

        xml_path = Path(__file__).with_name("assets") / "pusher_pos.xml"
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            str(xml_path),
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
        )
        utils.EzPickle.__init__(
            self,
            render_mode=render_mode,
            camera_configuration=camera_configuration,
            eef_step_scale=eef_step_scale,
            workspace_low=workspace_low,
            workspace_high=workspace_high,
            goal_distance_threshold=goal_distance_threshold,
            goal_hold_steps=goal_hold_steps,
            ik_damping=ik_damping,
            ik_step_scale=ik_step_scale,
            ik_max_joint_delta=ik_max_joint_delta,
            ik_iters=ik_iters,
            reward_near_weight=reward_near_weight,
            reward_dist_weight=reward_dist_weight,
            reward_control_weight=reward_control_weight,
            frame_skip=frame_skip,
            width=width,
            height=height,
            **kwargs,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.ik = PusherDeltaEEFIK(
            self,
            damping=ik_damping,
            step_scale=ik_step_scale,
            max_joint_delta=ik_max_joint_delta,
            iters=ik_iters,
        )
        self._apply_camera_configuration()

    def set_camera_configuration(self, camera_configuration: dict[str, Any] | None) -> None:
        self.camera_configuration = self.camera_configurator.prepare(camera_configuration)
        self._apply_camera_configuration()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._apply_camera_configuration()
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info.update(self._info())
        return obs, info

    def reset_model(self):
        self.termination.reset()
        qpos = self.init_qpos.copy()
        goal_xy = np.asarray([0.0, 0.0], dtype=np.float64)
        while True:
            object_xy = np.concatenate(
                [
                    self.np_random.uniform(low=-0.3, high=0.0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1),
                ]
            )
            if float(np.linalg.norm(object_xy - goal_xy)) > 0.17:
                break
        qpos[-4:-2] = object_xy
        qpos[-2:] = goal_xy
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)
        self.target_eef_xyz = self.get_body_com(TIP_BODY).astype(np.float64)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_ctrl = self.data.qpos[:ARM_JOINT_COUNT].astype(np.float32)
        self.last_target_clamped = False
        return self._get_obs()

    def step(self, action):
        self._apply_camera_configuration()
        array = np.asarray(action, dtype=np.float32).reshape(3)
        array = np.clip(array, -1.0, 1.0)
        if self.target_eef_xyz is None:
            self.target_eef_xyz = self.get_body_com(TIP_BODY).astype(np.float64)
        requested_target = (
            self.target_eef_xyz + array.astype(np.float64) * self.eef_step_scale
        )
        target_xyz = np.clip(
            requested_target,
            self.workspace_low,
            self.workspace_high,
        )
        clamped = bool(np.any(np.abs(target_xyz - requested_target) > 1e-12))
        ctrl = self.ik.solve_xyz(target_xyz)
        self.target_eef_xyz = target_xyz
        self.last_target_clamped = clamped
        self.last_action = array.copy()
        self.last_ctrl = ctrl.astype(np.float32)
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        reward, reward_info = self._reward(ctrl)
        reached, terminated = self.termination.update(self._object_goal_distance())
        info = self._info()
        info.update(reward_info)
        info["goal_termination"] = reached
        return obs, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:ARM_JOINT_COUNT],
                self.data.qvel.flatten()[:ARM_JOINT_COUNT],
                self.get_body_com(TIP_BODY),
                self.get_body_com(OBJECT_BODY),
                self.get_body_com(GOAL_BODY),
            ]
        ).astype(np.float64)

    def _reward(self, ctrl: np.ndarray) -> tuple[float, dict[str, float]]:
        fingertip_object = self._fingertip_object_distance()
        object_goal = self._object_goal_distance()
        reward_near = -fingertip_object * self.reward_near_weight
        reward_dist = -object_goal * self.reward_dist_weight
        reward_ctrl = -float(np.square(ctrl - self.data.qpos[:ARM_JOINT_COUNT]).sum()) * self.reward_control_weight
        return reward_near + reward_dist + reward_ctrl, {
            "reward_near": reward_near,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
        }

    def _object_goal_distance(self) -> float:
        return float(np.linalg.norm(self.get_body_com(OBJECT_BODY) - self.get_body_com(GOAL_BODY)))

    def _fingertip_object_distance(self) -> float:
        return float(np.linalg.norm(self.get_body_com(TIP_BODY) - self.get_body_com(OBJECT_BODY)))

    def _info(self) -> dict[str, Any]:
        object_goal = self._object_goal_distance()
        fingertip_object = self._fingertip_object_distance()
        return {
            "object_goal_distance": object_goal,
            "fingertip_object_distance": fingertip_object,
            "distance_to_target": object_goal,
            "goal_hold_counter": self.termination.counter,
            "fingertip_position": self.get_body_com(TIP_BODY).astype(np.float32),
            "object_position": self.get_body_com(OBJECT_BODY).astype(np.float32),
            "goal_position": self.get_body_com(GOAL_BODY).astype(np.float32),
            "ee_target_xyz": np.asarray(self.target_eef_xyz if self.target_eef_xyz is not None else [0.0, 0.0, 0.0], dtype=np.float32),
            "ee_target_reachability_clamped": bool(self.last_target_clamped),
            "last_action": self.last_action.astype(np.float32),
            "last_joint_position_target": self.last_ctrl.astype(np.float32),
        }

    def _apply_camera_configuration(self) -> None:
        if hasattr(self, "mujoco_renderer"):
            self.camera_configurator.apply(self.mujoco_renderer, self.camera_configuration)
