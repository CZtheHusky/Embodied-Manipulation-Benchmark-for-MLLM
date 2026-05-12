from __future__ import annotations

from typing import Any

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import Observation, Pose, SimAction, SimulatorAction, StepResult
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.utils.math import add_vec3, clamp, distance


class DummySimAdapter(SimAdapter):
    """Small deterministic kinematic simulator for pipeline tests."""

    def __init__(self, config: SimulatorConfig) -> None:
        if config.type != "dummy":
            raise ValueError(f"Unsupported simulator type: {config.type}")
        self.config = config
        self.step_id = 0
        self.ee_pose = Pose((0.0, 0.0, 0.0))
        self.target_position = (0.0, 0.0, 0.0)
        self.gripper = 0.0

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self.step_id = 0
        self.ee_pose = Pose(task_config.initial_ee_position)
        self.target_position = task_config.target_position
        self.gripper = 0.0
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        if not isinstance(action, SimAction):
            raise TypeError(f"DummySimAdapter requires SimAction, got {type(action).__name__}")
        self.step_id += 1
        raw_pos = add_vec3(self.ee_pose.position, action.delta_position)
        low = self.config.workspace_low
        high = self.config.workspace_high
        next_pos = (
            clamp(raw_pos[0], low[0], high[0]),
            clamp(raw_pos[1], low[1], high[1]),
            clamp(raw_pos[2], low[2], high[2]),
        )
        next_rot = add_vec3(self.ee_pose.rotation_rpy, action.delta_rotation_rpy)
        self.ee_pose = Pose(next_pos, next_rot)
        self.gripper = clamp(action.gripper, 0.0, 1.0)
        return StepResult(observation=self._observation(), terminated=False, truncated=False, info=self.get_state())

    def render(self, camera: CameraSpec) -> Any:
        return {
            "type": "dummy_rgb",
            "camera": camera.name,
            "width": camera.width,
            "height": camera.height,
            "ee_position": self.ee_pose.position,
            "target_position": self.target_position,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "ee_position": self.ee_pose.position,
            "ee_rotation_rpy": self.ee_pose.rotation_rpy,
            "target_position": self.target_position,
            "gripper": self.gripper,
            "distance_to_target": distance(self.ee_pose.position, self.target_position),
            "step_id": self.step_id,
        }

    def _observation(self) -> Observation:
        return Observation(step_id=self.step_id, images={}, state=self.get_state(), metadata={"simulator": "dummy"})
