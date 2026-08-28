from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.libero_controller import LiberoDeltaEEFController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    build_task,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import LiberoAction, Observation
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.libero.adapter import LiberoSimAdapter
from mllm_teleop_benchmark.ui.runtime import MujocoPlayerRuntimeHooks, hooks_for_config


MLLM_CONFIGS = (
    (
        "configs/libero_spatial_task0_mllm_chat_completions.yaml",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/libero_spatial_task0_mllm_chat_completions_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)
PUBLIC_CAMERAS = ("agentview", "robot0_eye_in_hand")
TASK_TEXT = (
    "pick up the black bowl between the plate and the ramekin and place it on "
    "the plate"
)
PRIVATE_SENTINELS = {
    "state": "PRIVATE_LIBERO_STATE_SENTINEL",
    "reward": "PRIVATE_LIBERO_REWARD_SENTINEL",
    "success": "PRIVATE_LIBERO_SUCCESS_SENTINEL",
    "object_pose": "PRIVATE_LIBERO_OBJECT_POSE_SENTINEL",
    "eef_pose": "PRIVATE_LIBERO_EEF_POSE_SENTINEL",
    "native_action": "PRIVATE_LIBERO_NATIVE_ACTION_SENTINEL",
    "seed": "PRIVATE_LIBERO_SEED_SENTINEL",
    "goal_state": "PRIVATE_LIBERO_GOAL_STATE_SENTINEL",
}


class LiberoFakeClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    "joystick_0": 1,
                    "joystick_1": 0,
                    "joystick_2": -1,
                    "joystick_3": 0,
                    "joystick_4": 1,
                    "joystick_5": 0,
                    "joystick_6": 0,
                },
                "Reflection": (
                    "The visible end-effector motion is being used to infer the "
                    "joystick controls."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-libero-{len(self.requests)}",
            "model": "fake-libero-model",
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


class _FakeCameraManager:
    def __init__(self, cameras=()) -> None:
        self._cameras = tuple(cameras)

    def get_active_cameras(self):
        return self._cameras

    def metadata(self):
        return {"manager": "fixed"}


def _import_runtime_dependency(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_LIBERO") == "1":
            raise
        pytest.skip(f"LIBERO runtime dependency unavailable: {module_name}: {exc}")


def test_libero_config_builds_without_libero_install():
    config = load_config("configs/libero_spatial_task0.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)
    sim = build_sim(config)
    task = build_task(config)
    provider = build_observation_provider(config, camera_manager=_FakeCameraManager())

    assert config.simulator.type == "libero"
    assert config.task.type == "libero_generic"
    assert tuple(action_space.channels) == tuple(f"joystick_{idx}" for idx in range(7))
    assert controller.metadata()["type"] == "libero_delta_eef"
    assert isinstance(sim, LiberoSimAdapter)
    assert task.instruction() == "pick up the black bowl between the plate and the ramekin and place it on the plate"
    assert provider.__class__.__name__ == "LiberoObservationProvider"


def test_libero_passes_mixed_external_and_hand_camera_sizes_to_native_env():
    config = load_config("configs/libero_spatial_task0.yaml")
    sim = build_sim(config)

    sim.set_camera_configuration(config.camera.cameras)

    assert sim._camera_dimensions() == ([512, 128], [512, 128])


def test_libero_config_uses_worker_play_runtime():
    config = load_config("configs/libero_spatial_task0.yaml")

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "agentview"
    assert isinstance(hooks_for_config(config), MujocoPlayerRuntimeHooks)


def test_libero_delta_eef_controller_outputs_7d_action():
    config = load_config("configs/libero_spatial_task0.yaml")
    controller = build_controller(config, build_mapping(config, config.benchmark.seed), config.benchmark.seed)
    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": -1.0,
                "joystick_2": 0.5,
                "joystick_3": 1.0,
                "joystick_4": 0.0,
                "joystick_5": -1.0,
                "joystick_6": 1.0,
            }
        )
    )

    assert isinstance(sim_action, LiberoAction)
    assert len(sim_action.values) == 7
    np.testing.assert_allclose(
        sim_action.values,
        [
            0.2,
            -0.2,
            0.1,
            0.0628318530717959,
            0.0,
            -0.0628318530717959,
            1.0,
        ],
    )

    mllm_config = load_config("configs/libero_spatial_task0_mllm_chat_completions.yaml")
    mllm_controller = build_controller(
        mllm_config,
        build_mapping(mllm_config, mllm_config.benchmark.seed),
        mllm_config.benchmark.seed,
    )
    normalized_action = mllm_controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": 0.0,
                "joystick_2": 0.0,
                "joystick_3": 1.0,
                "joystick_4": 0.0,
                "joystick_5": 0.0,
                "joystick_6": 0.0,
            }
        )
    )

    # The upstream OSC applies the physical gains after normalized input.
    normalized_pose = np.asarray(normalized_action.values[:6])
    np.testing.assert_allclose(
        normalized_pose,
        [0.2, 0.0, 0.0, 0.0628318530717959, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        normalized_pose[:3]
        * mllm_config.control_interface.params["native_translation_gain_m"],
        [0.01, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        normalized_pose[3:]
        * mllm_config.control_interface.params["native_rotation_gain_rad"],
        [0.0314159265358979, 0.0, 0.0],
    )


def test_libero_adapter_extracts_rgb_images_and_validity_metadata():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)
    adapter.last_native_obs = {
        "agentview_image": np.ones((1, 4, 5, 4), dtype=np.float32),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
    }
    adapter._task_metadata = {
        "language_instruction": config.task.params["language_instruction"],
        "suite_name": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "no_language_baseline": False,
        "random_baseline": True,
    }

    images = adapter._extract_images(adapter.last_native_obs)
    state = adapter.get_state()

    assert images["agentview"].shape == (4, 5, 3)
    assert images["agentview"].dtype == np.uint8
    assert state["ee_position"] == [0.1, 0.2, 0.3]
    assert state["benchmark_validity"]["suite_name"] == "libero_spatial"
    assert state["benchmark_validity"]["random_baseline"] is True


@pytest.mark.parametrize("config_path", [path for path, *_ in MLLM_CONFIGS])
def test_libero_mllm_initial_state_is_deterministic_and_seed_diverse(config_path):
    config = load_config(config_path)
    adapter = LiberoSimAdapter(config.simulator)

    assert config.simulator.params["init_state_strategy"] == "seed_modulo"
    assert config.task.params["init_state_strategy"] == "seed_modulo"
    assert adapter._select_init_state(config.task, seed=7, init_state_count=50) == (
        7,
        "seed_modulo",
    )
    assert adapter._select_init_state(config.task, seed=8, init_state_count=50) == (
        8,
        "seed_modulo",
    )
    assert adapter._select_init_state(config.task, seed=57, init_state_count=50) == (
        7,
        "seed_modulo",
    )


def test_libero_mllm_task_rejects_successful_reset_before_model_call():
    config = load_config(MLLM_CONFIGS[0][0])
    task = build_task(config)

    class StubSim:
        def reset(self, _task_config, seed):
            return Observation(
                step_id=0,
                images={},
                state={"is_success": seed == 8},
            )

    with pytest.raises(RuntimeError, match="already successful immediately after reset"):
        task.reset(StubSim(), seed=8)

    observation = task.reset(StubSim(), seed=7)

    assert observation.state["is_success"] is False


def test_libero_human_play_config_keeps_explicit_fixed_initial_state():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)

    assert config.simulator.params["settling_steps"] == 5
    assert config.task.params["reject_initial_success"] is True
    assert adapter._select_init_state(config.task, seed=42, init_state_count=50) == (
        0,
        "fixed",
    )


def test_libero_adapter_uses_official_settling_steps_without_counting_policy_steps():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)

    class StubEnv:
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            return (
                {"settling_index": len(self.actions)},
                0.0,
                False,
                {"success": False},
            )

        def check_success(self):
            return False

    adapter.env = StubEnv()
    adapter.last_native_obs = {"settling_index": 0}

    adapter._settle_after_init_state()

    assert len(adapter.env.actions) == 5
    assert all(action.shape == (7,) for action in adapter.env.actions)
    assert all(np.count_nonzero(action) == 0 for action in adapter.env.actions)
    assert adapter.last_native_obs["settling_index"] == 5
    assert adapter.step_id == 0
    assert adapter.last_reward is None
    assert adapter.last_done is False


def test_libero_native_horizon_preserves_policy_budget_after_settling():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)

    env_args = adapter._env_args(config.task, "task.bddl")

    assert config.task.max_steps == 600
    assert config.simulator.params["horizon"] == 600
    assert env_args["horizon"] == 605


def test_libero_adapter_rejects_instruction_that_does_not_match_selected_task():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)
    task = type("StubTask", (), {"language": "different upstream instruction"})()

    with pytest.raises(ValueError, match="does not match"):
        adapter._validate_language_instruction(config.task, task)


def test_libero_adapter_flips_raw_opengl_images_exactly_once():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)
    raw = np.asarray(
        [
            [[0, 0, 255], [0, 0, 255]],
            [[255, 0, 0], [255, 0, 0]],
        ],
        dtype=np.uint8,
    )

    frame = adapter._extract_images({"agentview_image": raw})["agentview"]

    np.testing.assert_array_equal(frame[0], raw[1])
    np.testing.assert_array_equal(frame[1], raw[0])


def test_libero_render_fallback_flips_raw_opengl_image_exactly_once():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)
    raw = np.asarray(
        [
            [[0, 0, 255], [0, 0, 255]],
            [[255, 0, 0], [255, 0, 0]],
        ],
        dtype=np.uint8,
    )

    class StubRenderer:
        def render(self, **_kwargs):
            return raw

    class StubEnv:
        sim = StubRenderer()

    adapter.env = StubEnv()
    adapter.last_native_obs = {}

    frame = adapter.render(config.camera.cameras[0])

    np.testing.assert_array_equal(frame[0], raw[1])
    np.testing.assert_array_equal(frame[1], raw[0])


def test_libero_runtime_reset_init_state_step_render_smoke_when_installed():
    _import_runtime_dependency("libero")
    _import_runtime_dependency("robosuite")

    config = load_config("configs/libero_spatial_task0.yaml")
    sim = build_sim(config)
    task = build_task(config)
    provider = build_observation_provider(config, camera_manager=_FakeCameraManager(config.camera.cameras))
    provider.configure_sim(sim)
    try:
        try:
            obs = task.reset(sim, seed=7)
            result = sim.step(LiberoAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)))
            observation = provider.get_observation(sim)
            frame = observation.images.get("agentview")
            if frame is None:
                frame = sim.render(config.camera.cameras[0])
        except (ImportError, FileNotFoundError, RuntimeError, AssertionError) as exc:
            if os.environ.get("MLLM_TELEOP_REQUIRE_LIBERO") == "1":
                raise
            pytest.skip(f"LIBERO runtime unavailable in this optional-backend environment: {exc}")

        assert obs.step_id == 0
        assert sim.get_state()["benchmark_validity"]["init_state_id"] == 0
        assert result.observation.step_id == 1
        assert np.asarray(frame).shape == (512, 512, 3)
        assert np.asarray(observation.images["robot0_eye_in_hand"]).shape == (
            128,
            128,
            3,
        )
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_libero_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    action_schema = build_action_space(config).schema()

    assert config.task.params["language_instruction"] == TASK_TEXT
    assert config.task.params["random_baseline"] is False
    assert config.task.params["reject_initial_success"] is True
    assert config.simulator.params["settling_steps"] == 5
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.model.max_output_tokens == 192
    assert config.agent.history.action_window == 3
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == PUBLIC_CAMERAS
    assert config.logging.save_video is False

    interface = config.control_interface
    assert interface.active_dimensions == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
    )
    assert interface.gripper == "binary"
    assert interface.mapping_strategy == mapping_strategy
    assert interface.mapping_visible_to_agent is visible
    assert interface.frame_transform == frame_transform
    assert interface.frame_visible_to_agent is visible
    assert interface.control_frame_description == (
        "the LIBERO MuJoCo world Cartesian x, y, and z axes used by robosuite's "
        "OSC_POSE controller; translation deltas and right-hand axis-angle "
        "rotation-vector commands are applied in this world frame. A mapped "
        "magnitude-1 benchmark command produces +/-0.01 m translation or "
        "+/-0.0314159265358979 rad (1.8 degrees) rotation, and in upright "
        "agentview image right is approximately +world y while "
        "image up is approximately -0.628 world x +0.778 world z. "
        "robot0_eye_in_hand moves with the wrist, so it has no fixed world-axis "
        "projection"
    )
    assert interface.params == {
        "scale_mode": "normalized_robosuite_osc_pose_input",
        "control_frame": "mujoco_world",
        "rotation_convention": "right_hand_axis_angle_rotvec",
        "native_translation_gain_m": 0.05,
        "native_rotation_gain_rad": 0.5,
        "sign_compensation": False,
    }
    assert interface.gripper_open_command == 0.0
    assert interface.gripper_closed_command == 1.0
    assert interface.gripper_hold_command is None
    assert config.controller.frame == "world"
    assert config.controller.scales == {
        "dx": 0.2,
        "dy": 0.2,
        "dz": 0.2,
        "droll": 0.0628318530717959,
        "dpitch": 0.0628318530717959,
        "dyaw": 0.0628318530717959,
        "gripper": 1.0,
    }
    agentview = next(
        camera for camera in config.camera.cameras if camera.name == "agentview"
    )
    assert agentview.params["basis_frame"] == "mujoco_world"
    assert agentview.params["basis_approximate"] is True
    image_right = np.asarray(agentview.params["image_right_in_world"])
    image_up = np.asarray(agentview.params["image_up_in_world"])
    np.testing.assert_allclose(image_right, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(image_up, [-0.628, 0.0, 0.778])
    np.testing.assert_allclose(np.linalg.norm(image_right), 1.0, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(image_up), 1.0, atol=2e-3)
    np.testing.assert_allclose(np.dot(image_right, image_up), 0.0, atol=2e-3)

    assert tuple(action_schema) == tuple(f"joystick_{index}" for index in range(7))
    assert all(
        action_schema[f"joystick_{index}"]
        == {"kind": "continuous", "min": -30.0, "max": 30.0}
        for index in range(6)
    )
    assert action_schema["joystick_6"] == {
        "kind": "discrete",
        "values": [0.0, 1.0],
    }

    group = make_experiment_group(config)
    assert group is not None
    assert "prompt-mllm_joystick_chat_completions_v11" in group
    assert "__output-" not in group
    if visible:
        assert "mapping-permuted_sign_flipped" not in group
        assert "frame-random_rotation_3d" not in group
    else:
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_libero_mllm_conditions_use_identical_upstream_task_text():
    instructions = {
        load_config(path).task.params["language_instruction"]
        for path, *_ in MLLM_CONFIGS
    }

    assert instructions == {TASK_TEXT}


def test_libero_mllm_public_prompt_matches_normalized_osc_pose_contract():
    baseline = load_config("configs/libero_spatial_task0_mllm_chat_completions.yaml")
    prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text

    assert "active controllable dimensions are: dx, dy, dz, droll, dpitch, dyaw" in prompt
    assert "dx, dy, and dz denote end-effector translation" in prompt
    assert "droll, dpitch, and dyaw denote end-effector rotation" in prompt
    assert "LIBERO MuJoCo world Cartesian x, y, and z axes" in prompt
    assert "right-hand axis-angle rotation-vector commands are applied" in prompt
    assert "magnitude-1 benchmark command produces +/-0.01 m translation" in prompt
    assert "1.8 degrees" in prompt
    assert "image right is approximately +world y" in prompt
    assert "image up is approximately -0.628 world x +0.778 world z" in prompt
    assert "robot0_eye_in_hand moves with the wrist" in prompt
    assert "no fixed world-axis projection" in prompt
    assert "0 means open, 1 means closed" in prompt
    assert "joystick_0 -> dx" in prompt
    assert "joystick_5 -> dyaw" in prompt
    assert "joystick_6 -> gripper" in prompt
    assert "No additional control-frame rotation is applied" in prompt

    perturbed = load_config(
        "configs/libero_spatial_task0_mllm_chat_completions_perturbed.yaml"
    )
    metadata = control_interface_metadata(perturbed, perturbed.benchmark.seed)
    perturbed_prompt = ControlInstructionBuilder().build(
        metadata,
        build_action_space(perturbed).schema(),
    ).text

    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert "right-hand axis-angle rotation-vector commands are applied" in perturbed_prompt
    assert (
        "magnitude-1 benchmark command produces +/-0.01 m translation"
        in perturbed_prompt
    )
    assert "1.8 degrees" in perturbed_prompt
    assert "robot0_eye_in_hand moves with the wrist" in perturbed_prompt
    assert "no fixed world-axis projection" in perturbed_prompt
    assert (
        "Infer the hidden controls from visual observations and your own action "
        "history"
    ) in perturbed_prompt
    assert "joystick_0 ->" not in perturbed_prompt
    assert "mapping" not in metadata
    assert "mapping_strategy" not in metadata
    assert "frame_transform" not in metadata
    assert "rotation_matrix" not in str(metadata)
    assert "quaternion_xyzw" not in str(metadata)


def test_libero_random_rotation_preserves_direction_within_native_action_range():
    mapping = FixedJoystickMapping(
        channels={
            "joystick_0": "dx",
            "joystick_1": "dy",
            "joystick_2": "dz",
            "joystick_3": "droll",
            "joystick_4": "dpitch",
            "joystick_5": "dyaw",
            "joystick_6": "gripper",
        },
        signs={f"joystick_{index}": 1.0 for index in range(7)},
    )
    frame_transform = RotationControlFrameTransform(
        rotation_matrix=(
            (2**-0.5, -(2**-0.5), 0.0),
            (2**-0.5, 2**-0.5, 0.0),
            (0.0, 0.0, 1.0),
        ),
        transform_type="random_rotation_3d",
        visible_to_agent=False,
    )
    controller = LiberoDeltaEEFController(
        mapping=mapping,
        frame_transform=frame_transform,
        scales={
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "droll": 1.0,
            "dpitch": 1.0,
            "dyaw": 1.0,
        },
    )
    action = JoystickAction(
        {
            "joystick_0": 1.0,
            "joystick_1": -1.0,
            "joystick_2": 0.0,
            "joystick_3": 1.0,
            "joystick_4": -1.0,
            "joystick_5": 0.0,
            "joystick_6": 0.0,
        }
    )
    raw_position = np.asarray(
        frame_transform.transform_delta_position((1.0, -1.0, 0.0))
    )
    raw_rotation = np.asarray(
        frame_transform.transform_delta_rotation((1.0, -1.0, 0.0))
    )
    sim_action = controller.translate(action)
    bounded_position = np.asarray(sim_action.values[:3])
    bounded_rotation = np.asarray(sim_action.values[3:6])

    assert np.max(np.abs(raw_position)) > 1.0
    assert np.max(np.abs(raw_rotation)) > 1.0
    assert np.max(np.abs(np.asarray(sim_action.values))) <= 1.0
    np.testing.assert_allclose(
        np.cross(raw_position, bounded_position),
        np.zeros(3),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.cross(raw_rotation, bounded_rotation),
        np.zeros(3),
        atol=1e-7,
    )


def test_libero_normalized_xyz_commands_move_eef_along_world_axes():
    _import_runtime_dependency("libero")
    _import_runtime_dependency("robosuite")
    config = load_config("configs/libero_spatial_task0_mllm_chat_completions.yaml")
    sim = build_sim(config)
    task = build_task(config)
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    try:
        observation = task.reset(sim, seed=config.benchmark.seed)
        before = np.asarray(observation.state["eef_pos"], dtype=float)
        for axis in range(3):
            channels = {f"joystick_{index}": 0.0 for index in range(7)}
            channels[f"joystick_{axis}"] = 1.0
            result = sim.step(controller.translate(JoystickAction(channels)))
            after = np.asarray(result.observation.state["eef_pos"], dtype=float)
            assert after[axis] - before[axis] > 1e-4
            before = after
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_libero_fake_client_closes_four_step_public_rgb_loop(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    _import_runtime_dependency("libero")
    _import_runtime_dependency("robosuite")
    client = LiberoFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-libero-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-libero.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_LIBERO_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = LiberoSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    native_actions = []
    oracle_checks = []
    oracle_goal_states = []
    original_step = LiberoSimAdapter.step

    def inspected_step(self, action):
        native_action = np.asarray(action.values, dtype=np.float32)
        low, high = self.env.env.action_spec
        assert native_action.shape == (7,)
        assert np.all(np.isfinite(native_action))
        assert np.all(native_action >= np.asarray(low))
        assert np.all(native_action <= np.asarray(high))
        native_actions.append(native_action.copy())
        goal_state = self.env.env.parsed_problem["goal_state"]
        assert goal_state
        oracle_goal_states.append(tuple(tuple(item) for item in goal_state))
        result = original_step(self, action)
        adapter_success = self._check_success()
        public_success = self.env.check_success()
        upstream_success = self.env.env._check_success()
        oracle_checks.append(
            (adapter_success, public_success, upstream_success, result.terminated)
        )
        return result

    monkeypatch.setattr(LiberoSimAdapter, "get_state", poisoned_get_state)
    monkeypatch.setattr(LiberoSimAdapter, "step", inspected_step)
    config = load_config(
        config_path,
        (
            "task.max_steps=4",
            "simulator.params.horizon=4",
            f"logging.output_dir={tmp_path}",
            "logging.save_video=false",
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert episode.metrics["agent_steps"] == 4
    assert episode.metrics["simulator_substeps"] == 4
    assert len(client.requests) == 4
    assert len(native_actions) == 4
    assert len(oracle_checks) == 4
    assert len(oracle_goal_states) == 4
    assert len(set(oracle_goal_states)) == 1
    assert all(
        adapter_success == public_success == upstream_success == terminated
        for adapter_success, public_success, upstream_success, terminated in oracle_checks
    )
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
    assert len(assistant_history) == 3
    assert all(item["Reflection"] for item in assistant_history)
    assert all(
        set(item["channels"]) == {f"joystick_{index}" for index in range(7)}
        for item in assistant_history
    )

    request_text = _request_text(client.requests[-1])
    assert "Prompt version: mllm_joystick_chat_completions_v11" in request_text
    assert f"Task: {TASK_TEXT}" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "droll, dpitch, and dyaw denote end-effector rotation" in request_text
    assert "LIBERO MuJoCo world Cartesian x, y, and z axes" in request_text
    assert "right-hand axis-angle rotation-vector commands are applied" in request_text
    assert (
        "magnitude-1 benchmark command produces +/-0.01 m translation"
        in request_text
    )
    assert "1.8 degrees" in request_text
    assert "robot0_eye_in_hand moves with the wrist" in request_text
    assert "no fixed world-axis projection" in request_text
    assert "0 means open, 1 means closed" in request_text
    if visible:
        assert "joystick_0 -> dx" in request_text
        assert "joystick_5 -> dyaw" in request_text
        assert "joystick_6 -> gripper" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert (
            "hidden joystick mapping and control-frame transform remain fixed "
            "for the entire episode"
        ) in request_text
        assert "joystick_0 ->" not in request_text

    encoded_requests = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    for sentinel in (*PRIVATE_SENTINELS.values(), "PRIVATE_LIBERO_API_KEY"):
        assert sentinel not in encoded_requests
    lower_request_text = request_text.lower()
    for forbidden in (
        "robot0_eef_pos",
        "object-state",
        "reward",
        "success oracle",
        "native action",
        "simulator action",
        "episode seed",
        "goal_state",
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
        assert np.all(np.abs(values) <= 1.0)

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 8
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        expected_shape = (128, 128, 3) if path.name == "robot0_eye_in_hand.png" else (512, 512, 3)
        assert pixels.shape == expected_shape
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["step_budget_unit"] == "agent_command"
    assert manifest["max_agent_steps"] == 4
    assert manifest["required_simulator_steps"] == 120
    assert manifest["configured_simulator_step_horizon"] == 120
    assert manifest["model"]["reasoning_effort"] == "low"
    assert "action_output_mode" not in manifest["model"]
    assert manifest["model"]["prompt_version"] == "mllm_joystick_chat_completions_v11"
    assert manifest["observation"]["cameras"] == list(PUBLIC_CAMERAS)
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
            assert b"PRIVATE_LIBERO_API_KEY" not in artifact.read_bytes()


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
