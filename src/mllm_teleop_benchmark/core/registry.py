from __future__ import annotations

from mllm_teleop_benchmark.agents.base import Agent
from mllm_teleop_benchmark.agents.mllm_policy import MLLMPolicy
from mllm_teleop_benchmark.agents.random_policy import RandomPolicy
from mllm_teleop_benchmark.agents.scripted_policy import ScriptedPolicy
from mllm_teleop_benchmark.config.schema import CameraConfig
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.control.ee_delta_pose import EEDeltaPoseController
from mllm_teleop_benchmark.control.gymnasium_box import GymnasiumBoxActionController, ReacherDeltaEEFJoystickController
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping, JoystickMapping
from mllm_teleop_benchmark.observation.camera import CameraManager, FixedCameraManager, RandomizedMujocoCameraManager
from mllm_teleop_benchmark.observation.provider import DefaultObservationProvider, ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.dummy import DummySimAdapter
from mllm_teleop_benchmark.tasks.base import Task
from mllm_teleop_benchmark.tasks.gymnasium_reacher import GymnasiumReacherTask
from mllm_teleop_benchmark.tasks.reach_target import ReachTargetTask


def build_sim(config: BenchmarkConfig) -> SimAdapter:
    if config.simulator.type == "dummy":
        return DummySimAdapter(config.simulator)
    if config.simulator.type == "gymnasium":
        from mllm_teleop_benchmark.sim.gymnasium_envs.adapter import GymnasiumSimAdapter

        return GymnasiumSimAdapter(config.simulator)
    raise ValueError(f"Unsupported simulator type: {config.simulator.type}")


def build_task(config: BenchmarkConfig) -> Task:
    if config.task.type == "reach_target":
        return ReachTargetTask(config.task)
    if config.task.type == "gymnasium_reacher":
        return GymnasiumReacherTask(config.task)
    raise ValueError(f"Unsupported task type: {config.task.type}")


def build_agent(config: BenchmarkConfig, seed: int) -> Agent:
    if config.agent.type == "random":
        return RandomPolicy(seed)
    if config.agent.type == "scripted":
        return ScriptedPolicy()
    if config.agent.type == "mllm":
        return MLLMPolicy()
    raise ValueError(f"Unsupported agent type: {config.agent.type}")


def build_camera_manager(config: CameraConfig, seed: int) -> CameraManager:
    if config.manager == "fixed":
        manager: CameraManager = FixedCameraManager(config)
    elif config.manager == "mujoco_randomized":
        manager = RandomizedMujocoCameraManager(config)
    else:
        raise ValueError(f"Unsupported camera manager: {config.manager}")
    manager.reset(seed)
    return manager


def build_observation_provider(config: BenchmarkConfig, camera_manager: CameraManager) -> ObservationProvider:
    if config.simulator.type == "gymnasium":
        from mllm_teleop_benchmark.sim.gymnasium_envs.observation import GymnasiumObservationProvider

        return GymnasiumObservationProvider(camera_manager)
    return DefaultObservationProvider(camera_manager)


def build_mapping(config: BenchmarkConfig) -> JoystickMapping:
    if config.mapping.type == "fixed":
        return FixedJoystickMapping.from_config(config.mapping)
    raise ValueError(f"Unsupported mapping type: {config.mapping.type}")


def build_controller(config: BenchmarkConfig, mapping: JoystickMapping) -> Controller:
    if config.controller.type == "ee_delta_pose":
        frame_transform = IdentityControlFrameTransform.from_config(config.frame_transform)
        return EEDeltaPoseController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "gymnasium_box":
        return GymnasiumBoxActionController.from_config(config.controller, mapping)
    if config.controller.type == "reacher_delta_eef":
        return ReacherDeltaEEFJoystickController.from_config(config.controller, mapping)
    raise ValueError(f"Unsupported controller type: {config.controller.type}")
