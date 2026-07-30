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
        if bool(config.params.get("visualize_reward", False)):
            raise ValueError(
                "DMControl visualize_reward must remain false because it encodes "
                "privileged reward information in public RGB observations"
            )

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
        raw_action_repeat = config.params.get("action_repeat", 1)
        if (
            isinstance(raw_action_repeat, bool)
            or not isinstance(raw_action_repeat, int)
            or raw_action_repeat < 1
        ):
            raise ValueError(
                "DMControl simulator.params.action_repeat must be a positive integer"
            )
        self.action_repeat = raw_action_repeat
        task_kwargs = dict(config.env_kwargs)
        task_kwargs["time_limit"] = float("inf")
        environment_kwargs = dict(config.params.get("environment_kwargs", {}))
        environment_kwargs.pop("time_limit", None)
        self.env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=False,
        )

        self.step_id = 0
        self.native_step_id = 0
        self.last_native_steps = 0
        self.last_timestep = None
        self.last_render = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self.step_id = 0
        self.native_step_id = 0
        self.last_native_steps = 0
        if hasattr(self.env.task, "random"):
            self.env.task.random.seed(seed)

        self.last_timestep = self.env.reset()
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if isinstance(action, DMControlPlanarEEFAction):
            planar_target = self._planar_eef_target(action)
            native_action = None
        elif isinstance(action, DMControlAction):
            native_action = action_to_native(action, self.env.action_spec(), clip=True)
            planar_target = None
        else:
            raise TypeError(
                "DMControlSimAdapter requires DMControlAction or "
                f"DMControlPlanarEEFAction, got {type(action).__name__}"
            )

        native_steps = 0
        for _ in range(self.action_repeat):
            if planar_target is not None:
                native_action = self._planar_eef_target_to_native(
                    planar_target,
                    action.gripper,
                )
            self.last_timestep = self.env.step(native_action)
            native_steps += 1
            if bool(self.last_timestep.last()):
                break

        self.step_id += 1
        self.native_step_id += native_steps
        self.last_native_steps = native_steps

        terminated, truncated = self._termination_flags(self.last_timestep)

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
            "native_step_id": self.native_step_id,
            "native_steps_this_action": self.last_native_steps,
            "action_repeat": self.action_repeat,
            "reward": self.last_timestep.reward if self.last_timestep else None,
            "current_reward": self._jsonable_array(
                self.env.task.get_reward(self.env.physics)
            ),
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
        return self._planar_eef_target_to_native(
            self._planar_eef_target(action),
            action.gripper,
        )

    def _planar_eef_target(self, action: DMControlPlanarEEFAction) -> Any:
        if self.domain_name != "manipulator":
            raise ValueError(
                "DMControlPlanarEEFAction is only supported for the manipulator domain"
            )

        import numpy as np

        delta = np.asarray(action.delta_xy, dtype=np.float64)
        if delta.shape != (2,) or not np.all(np.isfinite(delta)):
            raise ValueError("DMControl planar EEF delta must contain two finite values")
        if not np.isfinite(action.gripper) or not 0.0 <= action.gripper <= 1.0:
            raise ValueError("DMControl planar EEF gripper must be within [0, 1]")

        physics = self.env.physics
        model = physics.model
        data = physics.data
        grasp_site_id = model.name2id("grasp", "site")
        current_position = np.asarray(
            data.site_xpos[grasp_site_id, [0, 2]],
            dtype=np.float64,
        )
        return current_position + delta

    def _planar_eef_target_to_native(
        self,
        target_position: Any,
        gripper: float,
    ) -> Any:
        import mujoco
        import numpy as np

        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError("DMControl planar EEF target must contain two finite values")
        if not np.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
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
        current_position = np.asarray(
            data.site_xpos[grasp_site_id, [0, 2]],
            dtype=np.float64,
        )
        cartesian_force = (
            self.planar_eef_position_gain * (target - current_position)
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
        native[grasp_actuator_id] = 2.0 * float(gripper) - 1.0
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

    @staticmethod
    def _termination_flags(timestep: Any) -> tuple[bool, bool]:
        if not bool(timestep.last()):
            return False, False
        discount = getattr(timestep, "discount", None)
        if discount is not None and float(discount) > 0.0:
            return False, True
        return True, False

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
