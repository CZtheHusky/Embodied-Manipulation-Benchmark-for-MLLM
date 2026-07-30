from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import MagicMock

import multiprocessing as mp
import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    build_task,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import CalvinAction, Observation
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.sim.calvin.adapter import CalvinSimAdapter
from mllm_teleop_benchmark.ui.runtime import PlayerCommand, PlayerRuntimeResult, sim_adapter_worker_entry


MLLM_CONFIGS = (
    (
        "configs/calvin_delta_eef_mllm_reflection.yaml",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/calvin_delta_eef_mllm_reflection_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)

PRIVATE_SENTINELS = {
    "state": "PRIVATE_CALVIN_STATE_SENTINEL",
    "reward": "PRIVATE_CALVIN_REWARD_SENTINEL",
    "success_oracle": "PRIVATE_CALVIN_SUCCESS_SENTINEL",
    "distance": "PRIVATE_CALVIN_DISTANCE_SENTINEL",
    "object_pose": "PRIVATE_CALVIN_OBJECT_POSE_SENTINEL",
    "ee_pose": "PRIVATE_CALVIN_EEF_POSE_SENTINEL",
    "simulator_action": "PRIVATE_CALVIN_SIM_ACTION_SENTINEL",
    "seed": "PRIVATE_CALVIN_SEED_SENTINEL",
    "hidden_mapping_sample": "PRIVATE_CALVIN_MAPPING_SAMPLE_SENTINEL",
    "hidden_rotation_sample": "PRIVATE_CALVIN_ROTATION_SAMPLE_SENTINEL",
}


class CalvinFakeClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        active_channel = (len(self.requests) - 1) % 6
        channels = {
            f"joystick_{index}": 1 if index == active_channel else 0
            for index in range(6)
        }
        channels["joystick_6"] = 1
        text = json.dumps(
            {
                "channels": channels,
                "Reflection": (
                    "The RGB history is being used to identify the motion "
                    "channels while the gripper remains open."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-calvin-{len(self.requests)}",
            "model": "fake-calvin-model",
            "choices": [{"message": {"content": text}}],
        }
        return ModelResponse(
            text=text,
            request_id=raw["id"],
            model=raw["model"],
            usage=ModelUsage(input_tokens=100, output_tokens=32),
            raw=raw,
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


class _StaticResetSim:
    def __init__(self, state):
        self.state = state

    def reset(self, _task_config, _seed):
        return Observation(step_id=0, images={}, state=self.state, metadata={})


class _RecordingTasks:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filters = []

    def get_task_info_for_set(self, start_info, end_info, task_filter):
        self.filters.append(set(task_filter))
        return self.tasks.get_task_info_for_set(start_info, end_info, task_filter)


class _LegacyMovableObject:
    def __init__(self, np_random):
        self.np_random = np_random


class _LegacyScene:
    def __init__(self):
        self.np_random = np.random.default_rng(999)
        self.movable_objects = [
            _LegacyMovableObject(self.np_random),
            _LegacyMovableObject(self.np_random),
        ]


class _LegacySeedEnv:
    def __init__(self):
        self.np_random = np.random.default_rng(0)
        self.scene = _LegacyScene()
        self.layout = []

    @property
    def unwrapped(self):
        return self

    def seed(self, seed):
        self.np_random = np.random.default_rng(seed)

    def reset(self):
        self.layout = [
            movable_object.np_random.uniform(size=3).tolist()
            for movable_object in self.scene.movable_objects
        ]
        return {"scene_obs": self.layout}

    def get_info(self):
        return {"layout": self.layout}


def _contact(other_uid, other_link=-1):
    values = [0] * 5
    values[2] = other_uid
    values[4] = other_link
    return tuple(values)


def _oracle_info(
    *,
    red_position,
    red_contacts,
    robot_contacts,
    drawer_state=0.0,
):
    return {
        "robot_info": {
            "uid": 99,
            "contacts": list(robot_contacts),
        },
        "scene_info": {
            "doors": {
                "base__drawer": {
                    "current_state": drawer_state,
                }
            },
            "fixed_objects": {
                "table": {
                    "uid": 1,
                    "links": {
                        "base_link": 0,
                        "plank_link": 1,
                        "drawer_link": 2,
                    },
                }
            },
            "movable_objects": {
                "block_red": {
                    "uid": 10,
                    "current_pos": list(red_position),
                    "current_orn": [0.0, 0.0, 0.0, 1.0],
                    "current_lin_vel": [0.0, 0.0, 0.0],
                    "current_ang_vel": [0.0, 0.0, 0.0],
                    "contacts": list(red_contacts),
                }
            },
        },
    }


def test_calvin_config_load():
    """Verify that the CALVIN config can be loaded even if calvin-env is missing."""
    config = load_config("configs/calvin_delta_eef.yaml")
    assert config.simulator.type == "calvin"
    assert config.task.type == "calvin_generic"
    assert config.task.params["target_block_color"] == "red"
    assert config.task.params["instruction"] == (
        "Open the drawer, pick up the red block, and place the red block "
        "inside the drawer."
    )
    assert config.controller.type == "calvin_delta_eef"


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_calvin_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()

    assert config.task.params["target_block_color"] == "red"
    assert config.task.params["reject_initial_success"] is True
    assert config.task.params["instruction"] == (
        "Open the drawer, pick up the red block, and place the red block "
        "inside the drawer."
    )
    assert config.control_interface.active_dimensions == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
    )
    assert config.control_interface.gripper == "direct"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible
    control_description = config.control_interface.control_frame_description or ""
    assert "CALVIN's PyBullet world frame" in control_description
    assert "native magnitude 1 advances the target by 0.02 m" in control_description
    assert "PyBullet XYZ Euler parameters" in control_description
    assert "native magnitude 1 changes that parameter by 0.05 rad" in control_description
    assert "not an axis-angle vector" in control_description
    assert "gripper camera moves with the wrist" in control_description
    assert "values at or above 0.5 map to open" in control_description
    assert "values below 0.5 map to closed" in control_description
    if frame_transform == "random_rotation_3d":
        assert "hidden 3D frame transform applies only to translation" in control_description
        assert "uniformly rescaled by the largest absolute component" in control_description
        assert "preserves direction but can reduce step length" in control_description
    assert (
        config.control_interface.params["rotation_representation"]
        == "xyz_euler_parameters"
    )
    assert config.controller.scales == {
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "droll": 1.0,
        "dpitch": 1.0,
        "dyaw": 1.0,
    }
    assert config.controller.frame == "pybullet_world_xyz"
    assert config.controller.params["transform_rotation"] is False
    assert config.control_interface.gripper_open_command == 1.0
    assert config.control_interface.gripper_closed_command == 0.0
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.model.max_output_tokens == 192
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == ("static", "gripper")
    assert config.camera.manager == "fixed"
    assert config.camera.visible_metadata is False
    assert config.logging.save_video is False
    assert tuple(schema) == tuple(f"joystick_{index}" for index in range(7))
    assert all(
        schema[f"joystick_{index}"]
        == {"kind": "discrete", "values": [-1.0, 0.0, 1.0]}
        for index in range(6)
    )
    assert schema["joystick_6"] == {
        "kind": "continuous",
        "min": 0.0,
        "max": 1.0,
    }

    group = make_experiment_group(config)
    assert group is not None
    assert "output-action_with_reflection" in group
    if mapping_strategy == "fixed":
        assert "__mapping-" not in group
        assert "__frame-" not in group
    else:
        assert "__mapping-permuted_sign_flipped" in group
        assert "__frame-random_rotation_3d" in group


def test_calvin_mllm_task_instruction_is_identical_across_control_conditions():
    baseline = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    perturbed = load_config(
        "configs/calvin_delta_eef_mllm_reflection_perturbed.yaml"
    )

    assert baseline.task.params["instruction"] == perturbed.task.params["instruction"]


def test_calvin_task_uses_official_strict_three_stage_oracle():
    pytest.importorskip("calvin_env")
    config = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    task = build_task(config)
    reset_info = _oracle_info(
        red_position=(0.1, -0.1, 0.46),
        red_contacts=(_contact(1, 0),),
        robot_contacts=(),
    )
    opened_info = _oracle_info(
        red_position=(0.1, -0.1, 0.46),
        red_contacts=(_contact(1, 0),),
        robot_contacts=(),
        drawer_state=0.13,
    )
    held_info = _oracle_info(
        red_position=(0.1, -0.1, 0.53),
        red_contacts=(_contact(99),),
        robot_contacts=(_contact(10),),
        drawer_state=0.13,
    )
    placed_info = _oracle_info(
        red_position=(0.1, 0.0, 0.4),
        red_contacts=(_contact(1, 2),),
        robot_contacts=(),
        drawer_state=0.13,
    )

    observation = task.reset(_StaticResetSim(reset_info), seed=42)
    assert task._task_oracle.tasks["open_drawer"].args == (
        "base__drawer",
        0.12,
    )
    assert task._task_oracle.tasks["lift_red_block_table"].args == (
        "block_red",
        0.05,
        "table",
        "base_link",
    )
    assert task._task_oracle.tasks["place_in_drawer"].args == (
        "table",
        "drawer_link",
    )
    recording_tasks = _RecordingTasks(task._task_oracle)
    task._task_oracle = recording_tasks

    assert task.instruction() == (
        "Open the drawer, pick up the red block, and place the red block "
        "inside the drawer."
    )
    assert not task.is_success(observation, reset_info)
    assert not task.is_success(observation, opened_info)
    assert not task.is_success(observation, held_info)

    held_info["robot_info"]["contacts"].clear()
    held_info["scene_info"]["movable_objects"]["block_red"]["contacts"].clear()
    assert task.is_success(observation, placed_info)
    assert task.is_success(observation, {})
    assert recording_tasks.filters == [
        {"open_drawer"},
        {"open_drawer"},
        {"lift_red_block_table", "lift_red_block_slider"},
        {"place_in_drawer"},
    ]
    assert all("push_into_drawer" not in task_filter for task_filter in recording_tasks.filters)


def test_calvin_task_reset_invariant_accepts_valid_stub_initial_state():
    config = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    task = build_task(config)
    task._task_oracle = MagicMock()
    task._task_oracle.get_task_info_for_set.return_value = set()
    reset_info = _oracle_info(
        red_position=(0.1, -0.1, 0.46),
        red_contacts=(_contact(1, 0),),
        robot_contacts=(),
    )

    observation = task.reset(_StaticResetSim(reset_info), seed=42)

    assert task._reset_invariant_violations(reset_info) == ()
    assert not task.is_success(observation, observation.state)
    task._task_oracle.get_task_info_for_set.assert_called_once_with(
        task._reset_info,
        task._reset_info,
        {"open_drawer"},
    )


@pytest.mark.parametrize(
    ("reset_info", "message"),
    (
        (
            _oracle_info(
                red_position=(0.1, -0.1, 0.46),
                red_contacts=(_contact(1, 0),),
                robot_contacts=(),
                drawer_state=0.13,
            ),
            "drawer is not closed",
        ),
        (
            _oracle_info(
                red_position=(0.1, 0.0, 0.4),
                red_contacts=(_contact(1, 2),),
                robot_contacts=(),
            ),
            "red block is in contact with the drawer link",
        ),
        (
            _oracle_info(
                red_position=(0.1, -0.1, 0.53),
                red_contacts=(_contact(99),),
                robot_contacts=(_contact(10),),
            ),
            "red block is in contact with the robot",
        ),
    ),
)
def test_calvin_task_reset_invariant_rejects_invalid_stub_initial_state(
    reset_info,
    message,
):
    config = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    task = build_task(config)
    task._task_oracle = MagicMock()

    with pytest.raises(RuntimeError, match=message):
        task.reset(_StaticResetSim(deepcopy(reset_info)), seed=42)


def test_calvin_legacy_seed_fallback_syncs_scene_and_object_rngs():
    config = load_config("configs/calvin_delta_eef.yaml")
    sim = CalvinSimAdapter(config.simulator)
    env = _LegacySeedEnv()
    sim.env = env

    sim._reset_native(42)
    first = np.asarray(env.layout)
    assert env.scene.np_random is env.np_random
    assert all(
        movable_object.np_random is env.np_random
        for movable_object in env.scene.movable_objects
    )

    sim._reset_native(42)
    repeated = np.asarray(env.layout)
    sim._reset_native(43)
    different = np.asarray(env.layout)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, different)


def test_calvin_mllm_prompt_describes_public_pose_and_gripper_contract():
    baseline = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text
    baseline_prompt = prompt

    assert (
        "active controllable dimensions are: dx, dy, dz, droll, dpitch, dyaw"
        in prompt
    )
    assert "dx, dy, and dz denote end-effector translation" in prompt
    assert (
        "droll, dpitch, and dyaw denote increments of the target pose's "
        "x, y, and z Euler parameters"
    ) in prompt
    assert "positive mapped value commands the positive direction" in prompt
    assert "CALVIN's PyBullet world frame" in prompt
    assert "PyBullet XYZ Euler parameters" in prompt
    assert "not an axis-angle vector" in prompt
    assert "native magnitude 1 advances the target by 0.02 m" in prompt
    assert "joystick_0 -> dx" in prompt
    assert "joystick_5 -> dyaw" in prompt
    assert "joystick_6 -> gripper" in prompt
    assert "1 means open, 0 means closed" in prompt
    assert "values at or above 0.5 map to open" in prompt
    assert "No additional control-frame rotation is applied" in prompt

    perturbed = load_config(
        "configs/calvin_delta_eef_mllm_reflection_perturbed.yaml"
    )
    metadata = control_interface_metadata(perturbed, perturbed.benchmark.seed)
    prompt = ControlInstructionBuilder().build(
        metadata,
        build_action_space(perturbed).schema(),
    ).text

    assert "exact joystick-to-dimension mapping is not disclosed" in prompt
    assert "exact physical-frame transform is not disclosed" in prompt
    assert (
        "hidden joystick mapping and control-frame transform remain fixed "
        "for the entire episode"
    ) in prompt
    assert (
        "Infer the hidden controls from visual observations and your own action "
        "history"
    ) in prompt
    assert "no worked example or sampled hidden-control value is provided" in prompt
    assert "uniformly rescaled by the largest absolute component" in prompt
    assert "preserves direction but can reduce step length" in prompt
    assert "joystick_0 ->" not in prompt
    assert "mapping" not in metadata
    assert "mapping_strategy" not in metadata
    assert "frame_transform" not in metadata
    assert "rotation_matrix" not in str(metadata)

    for public_prompt in (baseline_prompt, prompt):
        assert (
            "active controllable dimensions are: dx, dy, dz, droll, dpitch, dyaw"
            in public_prompt
        )
        assert "The gripper is always the final virtual joystick" in public_prompt
        assert "1 means open, 0 means closed" in public_prompt
        for private_term in (
            "base__drawer",
            "drawer_link",
            "open_drawer",
            "place_in_drawer",
            "reject_initial_success",
            "robot_info",
            "scene_info",
        ):
            assert private_term not in public_prompt


def test_calvin_controller_translation():
    """Verify that the CALVIN controller correctly translates joystick actions."""
    config = load_config("configs/calvin_delta_eef.yaml")
    mapping = MagicMock()
    mapping.map_action.return_value = {
        "dx": 1.0,
        "dy": 0.0,
        "dz": -1.0,
        "droll": 0.0,
        "dpitch": 0.5,
        "dyaw": 0.0,
        "gripper": 1.0,
    }
    mapping.metadata.return_value = {}

    from mllm_teleop_benchmark.control.calvin_controller import CalvinDeltaEEFController

    controller = CalvinDeltaEEFController.from_config(config.controller, mapping)

    action = JoystickAction(channels={"joystick_0": 1.0})
    sim_action = controller.translate(action)

    assert isinstance(sim_action, CalvinAction)
    expected_values = [0.1, 0.0, -0.1, 0.0, 0.05, 0.0, 1.0]
    np.testing.assert_allclose(sim_action.values, expected_values)

    mapping.map_action.return_value = {
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "droll": 0.0,
        "dpitch": 0.0,
        "dyaw": 0.0,
        "gripper": 0.25,
    }
    sim_action = controller.translate(action)
    assert sim_action.values[-1] == -1.0

    mapping.map_action.return_value["gripper"] = 0.5
    sim_action = controller.translate(action)
    assert sim_action.values[-1] == 1.0


def test_calvin_controller_applies_frame_rotation_to_position_and_rotation():
    config = load_config("configs/calvin_delta_eef.yaml")
    mapping = MagicMock()
    mapping.map_action.return_value = {
        "dx": 1.0,
        "dy": 0.0,
        "dz": 0.0,
        "droll": 1.0,
        "dpitch": 0.0,
        "dyaw": 0.0,
        "gripper": 1.0,
    }
    mapping.metadata.return_value = {}

    from mllm_teleop_benchmark.control.calvin_controller import CalvinDeltaEEFController

    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = CalvinDeltaEEFController.from_config(config.controller, mapping, transform)

    sim_action = controller.translate(JoystickAction(channels={"joystick_0": 1.0}))

    np.testing.assert_allclose(sim_action.values[:3], [0.0, 0.1, 0.0], atol=1e-12)
    assert abs(sim_action.values[3]) < 1e-12


def test_calvin_mllm_controller_uses_native_scale_and_does_not_rotate_euler_parameters():
    config = load_config(
        "configs/calvin_delta_eef_mllm_reflection_perturbed.yaml"
    )
    mapping = MagicMock()
    mapping.map_action.return_value = {
        "dx": 1.0,
        "dy": 0.0,
        "dz": 0.0,
        "droll": 1.0,
        "dpitch": 0.0,
        "dyaw": 0.0,
        "gripper": 1.0,
    }
    mapping.metadata.return_value = {}

    from mllm_teleop_benchmark.control.calvin_controller import CalvinDeltaEEFController

    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(
            type="rotation_3d",
            visible_to_agent=False,
            params={"rpy_degrees": [0, 0, 90]},
        )
    )
    controller = CalvinDeltaEEFController.from_config(
        config.controller,
        mapping,
        transform,
    )
    sim_action = controller.translate(
        JoystickAction(channels={"joystick_0": 1.0})
    )

    np.testing.assert_allclose(sim_action.values[:3], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(sim_action.values[3:6], [1.0, 0.0, 0.0], atol=1e-12)
    assert controller.metadata()["transform_rotation"] is False


def test_calvin_hidden_rotation_preserves_direction_within_native_action_bounds():
    config = load_config(
        "configs/calvin_delta_eef_mllm_reflection_perturbed.yaml"
    )
    mapping = MagicMock()
    mapping.map_action.return_value = {
        "dx": 1.0,
        "dy": 1.0,
        "dz": 0.0,
        "droll": 0.0,
        "dpitch": 0.0,
        "dyaw": 0.0,
        "gripper": 1.0,
    }
    mapping.metadata.return_value = {}

    from mllm_teleop_benchmark.control.calvin_controller import CalvinDeltaEEFController

    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(
            type="rotation_3d",
            visible_to_agent=False,
            params={"rpy_degrees": [0, 0, 45]},
        )
    )
    controller = CalvinDeltaEEFController.from_config(
        config.controller,
        mapping,
        transform,
    )

    sim_action = controller.translate(
        JoystickAction(channels={"joystick_0": 1.0})
    )

    np.testing.assert_allclose(sim_action.values[:3], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.max(np.abs(sim_action.values[:3])) <= 1.0


def test_calvin_mllm_normalized_action_reaches_pinned_native_step_size():
    pytest.importorskip("calvin_env")
    from types import SimpleNamespace

    from calvin_env.robot.robot import Robot

    config = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    sim_action = controller.translate(
        JoystickAction(
            channels={
                "joystick_0": 1.0,
                "joystick_1": 0.0,
                "joystick_2": 0.0,
                "joystick_3": 0.0,
                "joystick_4": 0.0,
                "joystick_5": 0.0,
                "joystick_6": 1.0,
            }
        )
    )
    robot = SimpleNamespace(
        max_rel_pos=0.02,
        max_rel_orn=0.05,
        magic_scaling_factor_pos=1.0,
        magic_scaling_factor_orn=1.0,
        use_target_pose=True,
        target_pos=np.zeros(3),
        target_orn=np.zeros(3),
    )

    target_pos, target_orn, gripper = Robot.relative_to_absolute(
        robot,
        np.asarray(sim_action.values, dtype=float),
    )

    np.testing.assert_allclose(target_pos, [0.02, 0.0, 0.0])
    np.testing.assert_allclose(target_orn, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(gripper, [1.0])


def test_calvin_static_camera_projection_matches_public_prompt():
    calvin_env = pytest.importorskip("calvin_env")
    from omegaconf import OmegaConf

    camera_config = OmegaConf.load(
        Path(calvin_env.__file__).resolve().parents[1]
        / "conf"
        / "cameras"
        / "cameras"
        / "static.yaml"
    )
    eye = np.asarray(camera_config.look_from, dtype=float)
    target = np.asarray(camera_config.look_at, dtype=float)
    configured_up = np.asarray(camera_config.up_vector, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    image_right = np.cross(forward, configured_up)
    image_right /= np.linalg.norm(image_right)
    image_up = np.cross(image_right, forward)
    image_up /= np.linalg.norm(image_up)

    # +x is right, +y is upper-right, and +z is upper-left in the pinned view.
    assert image_right[0] > 0.7 and abs(image_up[0]) < 0.05
    assert image_right[1] > 0.5 and image_up[1] > 0.6
    assert image_right[2] < -0.4 and image_up[2] > 0.7


def test_calvin_perturbation_only_changes_motion_channels():
    config = load_config(
        "configs/calvin_delta_eef_mllm_reflection_perturbed.yaml"
    )
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller.frame_transform, RotationControlFrameTransform)
    assert set(mapping.channels.values()) == {
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
        "gripper",
    }
    assert mapping.channels["joystick_6"] == "gripper"
    assert mapping.signs["joystick_6"] == 1.0
    assert any(
        mapping.signs[f"joystick_{index}"] < 0.0 for index in range(6)
    )
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }
    assert controller.metadata()["transform_rotation"] is False


def test_calvin_sim_adapter_reset_step_smoke():
    """Run reset+step when real calvin_env is installed; otherwise skip cleanly."""
    pytest.importorskip("calvin_env")
    config = load_config("configs/calvin_delta_eef.yaml")
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    observation_provider.configure_sim(sim)
    try:
        obs = sim.reset(config.task, seed=42)
        assert obs.step_id == 0
        assert obs.state["step_id"] == 0
        obs = observation_provider.get_observation(sim)
        assert {"static", "gripper"}.issubset(obs.images)
        assert obs.images["static"].shape[-1] == 3
        assert obs.images["gripper"].shape[-1] == 3
        assert all(
            image.dtype == np.uint8
            and float(image.mean()) > 1.0
            and float(image.std()) > 1.0
            for image in obs.images.values()
        )
        rendered = np.asarray(sim.render(config.camera.cameras[0]))
        assert rendered.ndim == 3
        assert rendered.shape[-1] == 3
        assert rendered.dtype == np.uint8
        assert float(rendered.mean()) > 1.0
        assert float(rendered.std()) > 1.0

        native_action = np.asarray(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            dtype=np.float32,
        )
        assert native_action.shape == (7,)
        assert np.all(np.abs(native_action[:6]) <= 1.0)
        assert native_action[6] in {-1.0, 1.0}

        result = sim.step(CalvinAction(values=tuple(native_action)))
        assert result.observation.step_id == 1
        assert result.info["step_id"] == 1
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ((np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), "must be finite"),
        ((0.0, np.inf, 0.0, 0.0, 0.0, 0.0, 1.0), "must be finite"),
        ((1.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), "must be within"),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "must be exactly"),
    ),
)
def test_calvin_sim_adapter_rejects_invalid_native_actions(values, message):
    config = load_config("configs/calvin_delta_eef.yaml")
    sim = CalvinSimAdapter(config.simulator)
    sim.env = MagicMock()

    with pytest.raises(ValueError, match=message):
        sim.step(CalvinAction(values=values))

    sim.env.step.assert_not_called()


def test_calvin_real_seeded_resets_are_reproducible_and_distinct():
    pytest.importorskip("calvin_env")
    config = load_config("configs/calvin_delta_eef.yaml")
    sim = build_sim(config)

    def movable_positions(seed):
        state = sim.reset(config.task, seed=seed).state
        return {
            name: np.asarray(value["current_pos"], dtype=float)
            for name, value in state["scene_info"]["movable_objects"].items()
        }

    try:
        first = movable_positions(42)
        repeated = movable_positions(42)
        different = movable_positions(43)
    finally:
        sim.close()

    assert first.keys() == repeated.keys() == different.keys()
    assert all(
        np.allclose(first[name], repeated[name], rtol=0.0, atol=1e-3)
        for name in first
    )
    assert any(
        not np.allclose(first[name], different[name], rtol=0.0, atol=1e-2)
        for name in first
    )


def test_calvin_mllm_seed_range_reset_canary_when_installed():
    pytest.importorskip(
        "calvin_env",
        reason="pinned CALVIN reset canary requires calvin_env",
    )
    pytest.importorskip(
        "pybullet",
        reason="pinned CALVIN reset canary requires PyBullet",
    )
    pytest.importorskip(
        "hydra",
        reason="pinned CALVIN reset canary requires Hydra",
    )
    pytest.importorskip(
        "omegaconf",
        reason="pinned CALVIN reset canary requires OmegaConf",
    )
    config = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    sim = build_sim(config)
    task = build_task(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    observation_provider.configure_sim(sim)

    try:
        for seed in range(42, 52):
            observation = task.reset(sim, seed)
            reset_info = task._copy_oracle_info(observation.state)
            robot_info = reset_info["robot_info"]
            scene_info = reset_info["scene_info"]
            table_info = scene_info["fixed_objects"]["table"]
            red_info = scene_info["movable_objects"]["block_red"]
            drawer_contact = (
                table_info["uid"],
                table_info["links"]["drawer_link"],
            )

            assert (
                abs(scene_info["doors"]["base__drawer"]["current_state"])
                <= task._DRAWER_CLOSED_ATOL
            )
            assert drawer_contact not in {
                (contact[2], contact[4]) for contact in red_info["contacts"]
            }
            assert red_info["uid"] not in {
                contact[2] for contact in robot_info["contacts"]
            }
            assert robot_info["uid"] not in {
                contact[2] for contact in red_info["contacts"]
            }
            assert not task.is_success(observation, observation.state)

            public_observation = observation_provider.get_observation(sim)
            assert {"static", "gripper"}.issubset(public_observation.images)
            assert all(
                image.dtype == np.uint8
                and image.ndim == 3
                and image.shape[-1] == 3
                and float(image.mean()) > 1.0
                and float(image.std()) > 1.0
                for image in public_observation.images.values()
            )
    finally:
        sim.close()


def test_calvin_randomized_camera_manager():
    """Verify that RandomizedCalvinCameraManager samples positions."""
    from mllm_teleop_benchmark.config.schema import CameraConfig, CameraSpec
    from mllm_teleop_benchmark.observation.camera import RandomizedCalvinCameraManager

    config = CameraConfig(
        manager="calvin_randomized",
        visible_metadata=True,
        cameras=(
            CameraSpec(
                name="static",
                type="calvin_rgb",
                width=200,
                height=200,
                pose=(),
                params={
                    "position_ranges": ([0.5, 1.0], [-0.2, 0.2], [0.4, 0.8]),
                    "look_at_ranges": ([-0.1, 0.1], [-0.1, 0.1], [0.0, 0.2]),
                },
            ),
        ),
    )

    manager = RandomizedCalvinCameraManager(config)

    manager.reset(seed=42)
    active = manager.get_active_cameras()
    assert len(active) == 1
    cam = active[0]
    assert "position" in cam.params
    assert "look_at" in cam.params
    pos1 = cam.params["position"]

    manager.reset(seed=42)
    pos2 = manager.get_active_cameras()[0].params["position"]
    assert pos1 == pos2

    manager.reset(seed=43)
    pos3 = manager.get_active_cameras()[0].params["position"]
    assert pos1 != pos3


def test_calvin_sim_adapter_worker_reset_step_shutdown():
    pytest.importorskip("calvin_env")

    config = load_config("configs/calvin_delta_eef.yaml")
    context = mp.get_context("spawn")
    action_queue: mp.Queue = context.Queue()
    result_queue: mp.Queue = context.Queue()
    worker = context.Process(target=sim_adapter_worker_entry, args=(config, action_queue, result_queue))
    worker.start()
    try:
        action_queue.put(PlayerCommand("reset"))
        initial = result_queue.get(timeout=30)
        assert isinstance(initial, PlayerRuntimeResult)
        if initial.error:
            pytest.skip(f"CALVIN runtime unavailable: {initial.error}")
        assert initial.observation is not None
        assert np.asarray(initial.frame).ndim == 3

        controller = build_controller(config, build_mapping(config), config.benchmark.seed)
        sim_action = controller.translate(
            JoystickAction(
                {
                    "joystick_0": 0.0,
                    "joystick_1": 0.0,
                    "joystick_2": 0.0,
                    "joystick_3": 0.0,
                    "joystick_4": 0.0,
                    "joystick_5": 0.0,
                    "joystick_6": 1.0,
                }
            )
        )
        action_queue.put(PlayerCommand("step", sim_action))
        stepped = result_queue.get(timeout=30)
        assert isinstance(stepped, PlayerRuntimeResult)
        if stepped.error:
            pytest.skip(f"CALVIN runtime step unavailable: {stepped.error}")
        assert stepped.observation is not None
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1
    finally:
        action_queue.put(PlayerCommand("close"))
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_calvin_fake_client_closes_four_step_public_rgb_loop(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    pytest.importorskip("calvin_env")
    client = CalvinFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-calvin-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-calvin.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_CALVIN_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = CalvinSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    monkeypatch.setattr(CalvinSimAdapter, "get_state", poisoned_get_state)
    config = load_config(config_path)
    patched = replace(
        config,
        task=replace(config.task, max_steps=4),
        logging=replace(
            config.logging,
            output_dir=str(tmp_path),
            save_video=False,
        ),
    )

    episode = BenchmarkRunner(patched).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_image_count(request) for request in client.requests] == [2, 4, 6, 8]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]
    assistant_history = [
        json.loads(message["content"])
        for message in client.requests[-1].messages
        if message["role"] == "assistant"
    ]
    assert all(item["Reflection"] for item in assistant_history)
    assert all(
        set(item["channels"]) == {
            f"joystick_{index}" for index in range(7)
        }
        for item in assistant_history
    )

    request_text = _request_text(client.requests[-1])
    assert "Prompt version: mllm_joystick_reflection_v3" in request_text
    assert (
        "Open the drawer, pick up the red block, and place the red block "
        "inside the drawer."
        in request_text
    )
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert (
        "droll, dpitch, and dyaw denote increments of the target pose's "
        "x, y, and z Euler parameters"
    ) in request_text
    assert "CALVIN's PyBullet world frame" in request_text
    assert "PyBullet XYZ Euler parameters" in request_text
    assert "not an axis-angle vector" in request_text
    assert "native magnitude 1 advances the target by 0.02 m" in request_text
    assert "1 means open, 0 means closed" in request_text
    assert "values at or above 0.5 map to open" in request_text
    if mapping_strategy == "fixed":
        assert "joystick_0 -> dx" in request_text
        assert "joystick_5 -> dyaw" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert (
            "hidden joystick mapping and control-frame transform remain fixed "
            "for the entire episode"
        ) in request_text
        assert "uniformly rescaled by the largest absolute component" in request_text

    encoded_requests = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    for sentinel in (*PRIVATE_SENTINELS.values(), "PRIVATE_CALVIN_API_KEY"):
        assert sentinel not in encoded_requests
    lower_request_text = request_text.lower()
    for forbidden in (
        "simulator state",
        "reward",
        "success oracle",
        "distance",
        "object pose",
        "eef pose",
        "simulator action",
        "episode seed",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in lower_request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)
    for step in trajectory:
        values = np.asarray(step["sim_action"]["values"], dtype=float)
        assert values.shape == (7,)
        assert np.all(np.isfinite(values))
        assert np.all(np.abs(values[:6]) <= 1.000_001)
        assert values[6] in {-1.0, 1.0}

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 8
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.ndim == 3
        assert pixels.shape[-1] == 3
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v3"
    assert manifest["history"] == {
        "action_window": "all",
        "image_window": 3,
        "max_images_per_request": 128,
        "max_request_bytes": 67_108_864,
    }
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert manifest["control_interface"]["mapping_visible_to_agent"] is visible
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"PRIVATE_CALVIN_API_KEY" not in artifact.read_bytes()


def _image_count(request) -> int:
    return sum(
        1
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


def _request_text(request) -> str:
    values = []
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
            continue
        if not isinstance(content, list):
            continue
        values.extend(
            str(item["text"])
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return "\n".join(values)
