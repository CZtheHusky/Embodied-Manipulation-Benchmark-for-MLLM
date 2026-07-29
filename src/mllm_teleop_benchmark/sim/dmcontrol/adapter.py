from __future__ import annotations

import os
from typing import Any

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import (
    DMControlAction,
    DMControlPlanarEEFAction,
    Observation,
    SimulatorAction,
    StepResult,
)
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.dmcontrol.actions import action_to_native


class DMControlSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "dmcontrol":
            raise ValueError(f"Unsupported simulator type: {config.type}")

        os.environ.setdefault("MUJOCO_GL", str(config.params.get("mujoco_gl", "egl")))

        from dm_control import suite

        self.config = config
        domain_name = str(config.params.get("domain_name", ""))
        task_name = str(config.params.get("task_name", ""))
        if not domain_name or not task_name:
            raise ValueError("DMControl requires simulator.params.domain_name and simulator.params.task_name")

        self.domain_name = domain_name
        self.task_name = task_name
        self.planar_eef_position_gain = float(
            config.params.get("planar_eef_position_gain", 400.0)
        )
        self.planar_eef_velocity_damping = float(
            config.params.get("planar_eef_velocity_damping", 5.0)
        )
        self.env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=config.env_kwargs,
            visualize_reward=bool(config.params.get("visualize_reward", False)),
        )

        self.step_id = 0
        self.last_timestep = None
        self.last_render = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self.step_id = 0
        if hasattr(self.env.task, "random"):
            self.env.task.random.seed(seed)

        self.last_timestep = self.env.reset()
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if isinstance(action, DMControlPlanarEEFAction):
            native_action = self._planar_eef_to_native(action)
        elif isinstance(action, DMControlAction):
            native_action = action_to_native(action, self.env.action_spec(), clip=True)
        else:
            raise TypeError(
                "DMControlSimAdapter requires DMControlAction or "
                f"DMControlPlanarEEFAction, got {type(action).__name__}"
            )

        self.last_timestep = self.env.step(native_action)
        self.step_id += 1

        terminated = self.last_timestep.last()
        truncated = False

        return StepResult(
            observation=self._observation(),
            terminated=terminated,
            truncated=truncated,
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        width = camera.width
        height = camera.height
        camera_id = camera.params.get("camera_id", camera.name)

        frame = self.env.physics.render(height=height, width=width, camera_id=camera_id)
        self.last_render = frame
        return frame

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "reward": self.last_timestep.reward if self.last_timestep else None,
            "step_type": int(self.last_timestep.step_type) if self.last_timestep else None,
            "domain": self.domain_name,
            "task": self.task_name,
        }

        if self.last_timestep and self.last_timestep.observation:
            for key, value in self.last_timestep.observation.items():
                state[f"observation/{key}"] = self._jsonable_array(value)

        return state

    def close(self) -> None:
        self.env.close()

    def _planar_eef_to_native(self, action: DMControlPlanarEEFAction) -> Any:
        if self.domain_name != "manipulator":
            raise ValueError(
                "DMControlPlanarEEFAction is only supported for the manipulator domain"
            )

        import mujoco
        import numpy as np

        delta = np.asarray(action.delta_xy, dtype=np.float64)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("DMControl planar EEF delta must contain two finite values")
        if not np.isfinite(action.gripper) or not 0.0 <= action.gripper <= 1.0:
            raise ValueError("DMControl planar EEF gripper must be within [0, 1]")

        physics = self.env.physics
        model = physics.model
        data = physics.data
        arm_actuator_ids = np.asarray(
            [model.name2id(name, "actuator") for name in ("root", "shoulder", "elbow", "wrist")],
            dtype=np.int32,
        )
        arm_joint_ids = np.asarray(model.actuator_trnid[arm_actuator_ids, 0], dtype=np.int32)
        arm_dof_ids = np.asarray(model.jnt_dofadr[arm_joint_ids], dtype=np.int32)
        grasp_site_id = model.name2id("grasp", "site")

        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model.ptr, data.ptr, jacobian, None, grasp_site_id)
        planar_jacobian = jacobian[[0, 2]][:, arm_dof_ids]
        planar_velocity = planar_jacobian @ np.asarray(data.qvel[arm_dof_ids])
        cartesian_force = (
            self.planar_eef_position_gain * delta
            - self.planar_eef_velocity_damping * planar_velocity
        )
        joint_torque = (
            planar_jacobian.T @ cartesian_force
            + np.asarray(data.qfrc_bias[arm_dof_ids])
        )

        arm_gear = np.asarray(model.actuator_gear[arm_actuator_ids, 0])
        if np.any(np.isclose(arm_gear, 0.0)):
            raise ValueError("DMControl manipulator arm actuator gear must be non-zero")
        arm_control = joint_torque / arm_gear

        action_spec = self.env.action_spec()
        native = np.zeros(action_spec.shape, dtype=np.float64)
        native[arm_actuator_ids] = arm_control
        grasp_actuator_id = model.name2id("grasp", "actuator")
        native[grasp_actuator_id] = 2.0 * float(action.gripper) - 1.0
        return action_to_native(
            DMControlAction(values=tuple(float(value) for value in native)),
            action_spec,
            clip=True,
        )

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={
                "simulator": "dmcontrol",
                "domain": self.domain_name,
                "task": self.task_name,
            },
        )

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
