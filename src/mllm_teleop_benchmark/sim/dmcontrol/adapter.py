from __future__ import annotations

import os
from typing import Any

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import DMControlAction, Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.dmcontrol.actions import action_to_native


class DMControlSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "dmcontrol":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        
        # DMControl/MuJoCo GL context selection
        os.environ.setdefault("MUJOCO_GL", str(config.params.get("mujoco_gl", "egl")))
        
        from dm_control import suite
        import numpy as np
        
        self.config = config
        domain_name = config.params.get("domain_name")
        task_name = config.params.get("task_name")
        
        if not domain_name or not task_name:
            raise ValueError("DMControl requires 'domain_name' and 'task_name' in simulator.params")
        
        self.env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=config.env_kwargs,
            visualize_reward=config.params.get("visualize_reward", False)
        )
        
        self.step_id = 0
        self.last_timestep = None
        self.last_render = None

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self.step_id = 0
        import numpy as np
        # dm_control uses a random state for seeding if not using composer
        # suite.load environments don't have a direct reset(seed=...) like gymnasium v0.26+
        # But we can try to set the seed in the environment's random state if it exists
        if hasattr(self.env.task, "random"):
             self.env.task.random.seed(seed)
        
        self.last_timestep = self.env.reset()
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, DMControlAction):
            raise TypeError(f"DMControlSimAdapter requires DMControlAction, got {type(action).__name__}")
        
        native_action = action_to_native(action, self.env.action_spec(), clip=True)
        
        self.last_timestep = self.env.step(native_action)
        self.step_id += 1
        
        terminated = self.last_timestep.last()
        truncated = False # DMControl doesn't distinguish by default in TimeStep
        
        return StepResult(
            observation=self._observation(),
            terminated=terminated,
            truncated=truncated,
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        # camera.name can be a string ID or an integer
        width = camera.width
        height = camera.height
        camera_id = camera.params.get("camera_id", camera.name)
        
        # Use physics.render for DMControl
        frame = self.env.physics.render(height=height, width=width, camera_id=camera_id)
        self.last_render = frame
        return frame

    def get_state(self) -> dict[str, Any]:
        state = {
            "step_id": self.step_id,
            "reward": self.last_timestep.reward if self.last_timestep else None,
            "step_type": int(self.last_timestep.step_type) if self.last_timestep else None,
        }
        
        if self.last_timestep and self.last_timestep.observation:
            for key, value in self.last_timestep.observation.items():
                state[f"observation/{key}"] = self._jsonable_array(value)
                
        return state

    def close(self) -> None:
        self.env.close()

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={
                "simulator": "dmcontrol",
                "domain": self.config.params.get("domain_name"),
                "task": self.config.params.get("task_name")
            }
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
