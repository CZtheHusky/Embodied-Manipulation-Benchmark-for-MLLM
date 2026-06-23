from __future__ import annotations

from mllm_teleop_benchmark.agents.base import Agent
from mllm_teleop_benchmark.agents.mllm_policy import MLLMPolicy
from mllm_teleop_benchmark.agents.random_policy import RandomPolicy
from mllm_teleop_benchmark.agents.scripted_policy import ScriptedPolicy
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.config.schema import CameraConfig
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.control.ee_delta_pose import EEDeltaPoseController
from mllm_teleop_benchmark.control.gymnasium_box import GymnasiumBoxActionController, PusherDeltaEEFJoystickController, ReacherDeltaEEFJoystickController
from mllm_teleop_benchmark.control.maniskill import ManiSkillDeltaEEFController
from mllm_teleop_benchmark.mapping.frame_transform import ControlFrameTransform, IdentityControlFrameTransform, RotationControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping, JoystickMapping
from mllm_teleop_benchmark.observation.camera import CameraManager, FixedCameraManager, RandomizedManiSkillCameraManager, RandomizedMujocoCameraManager
from mllm_teleop_benchmark.observation.provider import DefaultObservationProvider, ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.sim.dummy import DummySimAdapter
from mllm_teleop_benchmark.tasks.base import Task
from mllm_teleop_benchmark.tasks.gymnasium_pusher import GymnasiumPusherTask
from mllm_teleop_benchmark.tasks.gymnasium_reacher import GymnasiumReacherTask
from mllm_teleop_benchmark.tasks.reach_target import ReachTargetTask


def build_frame_transform(config: BenchmarkConfig, seed: int | None = None) -> ControlFrameTransform:
    if config.frame_transform.type == "identity":
        return IdentityControlFrameTransform.from_config(config.frame_transform)
    if config.frame_transform.type in {"rotation_2d", "rotation_3d", "random_rotation_2d", "random_rotation_3d"}:
        return RotationControlFrameTransform.from_config(config.frame_transform, seed)
    raise ValueError(f"Unsupported frame transform type: {config.frame_transform.type}")


def build_sim(config: BenchmarkConfig) -> SimAdapter:
    if config.simulator.type == "dummy":
        return DummySimAdapter(config.simulator)
    if config.simulator.type == "gymnasium":
        from mllm_teleop_benchmark.sim.gymnasium_envs.adapter import GymnasiumSimAdapter

        return GymnasiumSimAdapter(config.simulator)
    if config.simulator.type == "maniskill":
        from mllm_teleop_benchmark.sim.maniskill.adapter import ManiSkillSimAdapter

        return ManiSkillSimAdapter(config.simulator)
    if config.simulator.type == "robosuite":
        from mllm_teleop_benchmark.sim.robosuite_adapter import RobosuiteSimAdapter

        return RobosuiteSimAdapter(config.simulator)
    if config.simulator.type == "dmcontrol":
        from mllm_teleop_benchmark.sim.dmcontrol.adapter import DMControlSimAdapter

        return DMControlSimAdapter(config.simulator)
    raise ValueError(f"Unsupported simulator type: {config.simulator.type}")


def build_task(config: BenchmarkConfig) -> Task:
    if config.task.type == "reach_target":
        return ReachTargetTask(config.task)
    if config.task.type == "gymnasium_reacher":
        return GymnasiumReacherTask(config.task)
    if config.task.type == "gymnasium_pusher":
        return GymnasiumPusherTask(config.task)
    if config.task.type == "maniskill_generic":
        from mllm_teleop_benchmark.tasks.maniskill import ManiSkillTask

        return ManiSkillTask(config.task)
    if config.task.type == "robosuite_lift":
        from mllm_teleop_benchmark.tasks.robosuite_task import RobosuiteLiftTask

        return RobosuiteLiftTask(config.task)
    if config.task.type == "dmcontrol_generic":
        from mllm_teleop_benchmark.tasks.dmcontrol_task import DMControlTask

        return DMControlTask(config.task)
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
    elif config.manager == "maniskill_randomized":
        manager = RandomizedManiSkillCameraManager(config)
    else:
        raise ValueError(f"Unsupported camera manager: {config.manager}")
    manager.reset(seed)
    return manager


def build_observation_provider(config: BenchmarkConfig, camera_manager: CameraManager) -> ObservationProvider:
    if config.simulator.type == "gymnasium":
        from mllm_teleop_benchmark.sim.gymnasium_envs.observation import GymnasiumObservationProvider

        return GymnasiumObservationProvider(camera_manager)
    if config.simulator.type == "maniskill":
        from mllm_teleop_benchmark.sim.maniskill.observation import ManiSkillObservationProvider

        return ManiSkillObservationProvider(camera_manager)
    if config.simulator.type == "dmcontrol":
        from mllm_teleop_benchmark.sim.dmcontrol.observation import DMControlObservationProvider

        return DMControlObservationProvider(camera_manager)
    return DefaultObservationProvider(camera_manager)


def build_mapping(config: BenchmarkConfig, seed: int | None = None) -> JoystickMapping:
    mapping_config = config.control_interface.to_mapping_config(seed)
    if mapping_config.type == "fixed":
        return FixedJoystickMapping.from_config(mapping_config)
    raise ValueError(f"Unsupported mapping type: {mapping_config.type}")


def build_action_space(config: BenchmarkConfig) -> ActionSpace:
    return ActionSpace.from_config(config.action_space)


def control_interface_metadata(config: BenchmarkConfig, seed: int | None = None) -> dict[str, object]:
    metadata = config.control_interface.metadata()
    metadata["source"] = "control_interface"
    metadata["mapping_complexity_present"] = config.control_interface.mapping_strategy != "fixed"
    metadata["frame_complexity_present"] = config.control_interface.frame_transform != "identity"
    return metadata


def build_controller(config: BenchmarkConfig, mapping: JoystickMapping, seed: int | None = None) -> Controller:
    frame_transform = build_frame_transform(config, seed)
    if config.controller.type == "ee_delta_pose":
        return EEDeltaPoseController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "gymnasium_box":
        return GymnasiumBoxActionController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "reacher_delta_eef":
        return ReacherDeltaEEFJoystickController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "pusher_delta_eef":
        return PusherDeltaEEFJoystickController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "maniskill_delta_eef":
        return ManiSkillDeltaEEFController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "robosuite_osc":
        from mllm_teleop_benchmark.control.robosuite_controller import RobosuiteOSCController

        return RobosuiteOSCController.from_config(config.controller, mapping, frame_transform)
    if config.controller.type == "dmcontrol":
        from mllm_teleop_benchmark.control.dmcontrol_controller import DMControlController

        return DMControlController.from_config(config.controller, mapping, frame_transform)
    raise ValueError(f"Unsupported controller type: {config.controller.type}")
