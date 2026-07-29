import json
import os
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.rlbench import RLBenchDeltaEEFController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import RLBenchAction
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.sim.rlbench.adapter import RLBenchSimAdapter
from mllm_teleop_benchmark.sim.rlbench.observation import RLBenchObservationProvider


ROOT = Path(__file__).resolve().parents[1]
MLLM_CONFIGS = (
    (
        "configs/rlbench_reach_target_mllm_reflection.yaml",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/rlbench_reach_target_mllm_reflection_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)
PRIVATE_SENTINELS = {
    "private_state": "PRIVATE_RLBENCH_STATE_SENTINEL",
    "private_reward": "PRIVATE_RLBENCH_REWARD_SENTINEL",
    "private_success": "PRIVATE_RLBENCH_SUCCESS_SENTINEL",
    "private_distance": "PRIVATE_RLBENCH_DISTANCE_SENTINEL",
    "private_target": "PRIVATE_RLBENCH_TARGET_SENTINEL",
    "private_eef_pose": "PRIVATE_RLBENCH_EEF_SENTINEL",
    "private_native_action": "PRIVATE_RLBENCH_NATIVE_ACTION_SENTINEL",
    "private_seed": "PRIVATE_RLBENCH_SEED_SENTINEL",
}


class RLBenchFakeClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    "joystick_0": 0,
                    "joystick_1": 0,
                    "joystick_2": 0,
                    "joystick_3": 0,
                },
                "Reflection": (
                    "The public RGB history will be used to identify the three "
                    "translation controls while the gripper remains open."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-rlbench-{len(self.requests)}",
            "model": "fake-rlbench-model",
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


def test_rlbench_uv_input_pins_python_source_and_excludes_coppeliasim():
    requirement = ROOT.joinpath("envs/uv/rlbench.in").read_text()

    assert re.search(r"^rlbench @ git\+https://github\.com/stepjam/RLBench\.git@[0-9a-f]{40}$", requirement, re.MULTILINE)
    assert "coppeliasim" not in requirement.lower()


def test_rlbench_reach_target_config_builds_without_rlbench_install():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)
    sim = build_sim(config)

    assert config.simulator.type == "rlbench"
    assert config.task.type == "rlbench_generic"
    assert config.controller.type == "rlbench_delta_eef"
    assert tuple(action_space.channels) == ("joystick_0", "joystick_1", "joystick_2")
    assert controller.metadata()["type"] == "rlbench_delta_eef"
    assert isinstance(sim, RLBenchSimAdapter)


def test_rlbench_configs_use_worker_play_runtime():
    for config_path in (
        "configs/rlbench_reach_target_delta_eef.yaml",
        "configs/rlbench_reach_target_delta_eef_smoke.yaml",
        *(item[0] for item in MLLM_CONFIGS),
    ):
        config = load_config(config_path)

        assert config.ui.backend == "pygame"
        assert config.ui.runtime == "sim_adapter_worker"
        assert config.ui.frame_camera == "front"


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_rlbench_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()

    assert config.simulator.params["headless"] is False
    assert config.task.params["instruction"] == (
        "Move the robot end-effector to the visible target."
    )
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.gripper == "binary"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible
    assert config.control_interface.control_frame_description == (
        "the RLBench world-frame x, y, and z axes used by the end-effector pose planner"
    )
    assert config.control_interface.gripper_open_command == 0.0
    assert config.control_interface.gripper_closed_command == 1.0
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.model.max_output_tokens == 192
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == ("front",)
    assert config.logging.save_video is False
    assert tuple(schema) == (
        "joystick_0",
        "joystick_1",
        "joystick_2",
        "joystick_3",
    )
    assert all(
        schema[f"joystick_{index}"]
        == {"kind": "discrete", "values": [-1.0, 0.0, 1.0]}
        for index in range(3)
    )
    assert schema["joystick_3"] == {
        "kind": "discrete",
        "values": [0.0, 1.0],
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


def test_rlbench_mllm_task_and_public_prompt_contract():
    baseline = load_config("configs/rlbench_reach_target_mllm_reflection.yaml")
    perturbed = load_config(
        "configs/rlbench_reach_target_mllm_reflection_perturbed.yaml"
    )
    assert baseline.task.params["instruction"] == perturbed.task.params["instruction"]
    assert baseline.benchmark.seed == perturbed.benchmark.seed
    assert baseline.simulator == perturbed.simulator
    assert baseline.task == perturbed.task
    assert baseline.camera == perturbed.camera
    assert baseline.controller == perturbed.controller
    assert baseline.agent == perturbed.agent

    baseline_prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text
    perturbed_prompt = ControlInstructionBuilder().build(
        control_interface_metadata(perturbed, perturbed.benchmark.seed),
        build_action_space(perturbed).schema(),
    ).text

    assert "active controllable dimensions are: dx, dy, dz" in baseline_prompt
    assert "dx, dy, and dz denote end-effector translation" in baseline_prompt
    assert "RLBench world-frame x, y, and z axes" in baseline_prompt
    assert "joystick_0 -> dx" in baseline_prompt
    assert "0 means open, 1 means closed" in baseline_prompt
    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert "fixed for the entire episode" in perturbed_prompt
    assert "permuted_sign_flipped" not in perturbed_prompt
    assert "random_rotation_3d" not in perturbed_prompt


def test_rlbench_mllm_gripper_endpoints_map_to_native_discrete_semantics():
    config = load_config("configs/rlbench_reach_target_mllm_reflection.yaml")
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    motion = {
        "joystick_0": 0.0,
        "joystick_1": 0.0,
        "joystick_2": 0.0,
    }

    open_action = controller.translate(
        JoystickAction({**motion, "joystick_3": 0.0})
    )
    closed_action = controller.translate(
        JoystickAction({**motion, "joystick_3": 1.0})
    )

    assert open_action.gripper == 1.0
    assert closed_action.gripper == 0.0


def test_rlbench_perturbation_keeps_gripper_channel_unmodified():
    config = load_config(
        "configs/rlbench_reach_target_mllm_reflection_perturbed.yaml"
    )
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert set(mapping.channels.values()) == {"dx", "dy", "dz", "gripper"}
    assert mapping.channels["joystick_3"] == "gripper"
    assert mapping.signs["joystick_3"] == 1.0
    assert isinstance(controller.frame_transform, RotationControlFrameTransform)
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }


def test_rlbench_delta_eef_controller_maps_delta_position_and_default_gripper():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    controller = RLBenchDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed))

    sim_action = controller.translate(JoystickAction({"joystick_0": 1.0, "joystick_1": -1.0, "joystick_2": 0.0}))

    assert isinstance(sim_action, RLBenchAction)
    assert sim_action.delta_position == (0.02, -0.02, 0.0)
    assert sim_action.delta_rotation_rpy == (0.0, 0.0, 0.0)
    assert sim_action.gripper == 1.0


def test_rlbench_delta_eef_controller_applies_frame_rotation():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = RLBenchDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(JoystickAction({"joystick_0": 1.0, "joystick_1": 0.0, "joystick_2": 0.0}))

    np.testing.assert_allclose(sim_action.delta_position, [0.0, 0.02, 0.0], atol=1e-12)


def test_rlbench_adapter_converts_delta_to_absolute_pose_action():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    adapter = RLBenchSimAdapter(config.simulator)
    adapter.last_native_obs = type(
        "FakeRLBenchObs",
        (),
        {"gripper_pose": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)},
    )()

    native = adapter._native_action(RLBenchAction(delta_position=(0.1, -0.1, 1.0), gripper=2.0))

    np.testing.assert_allclose(native, [0.1, -0.1, 1.35, 0.0, 0.0, 0.0, 1.0, 1.0])


def test_rlbench_observation_provider_extracts_configured_rgb_camera():
    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    provider = RLBenchObservationProvider(build_camera_manager(config.camera, config.benchmark.seed))
    frame = np.zeros((4, 5, 4), dtype=np.float32)
    frame[..., 0] = 1.0
    obs = type("FakeRLBenchObs", (), {"front_rgb": frame})()

    images = provider._extract_images(obs)

    assert images["front"].shape == (4, 5, 3)
    assert images["front"].dtype == np.uint8
    assert images["front"][0, 0, 0] == 255


def test_rlbench_runtime_reset_step_render_smoke_when_installed():
    try:
        __import__("rlbench")
        __import__("pyrep")
    except ImportError as exc:
        pytest.skip(f"RLBench/PyRep are not installed in this environment: {exc}")

    config = load_config("configs/rlbench_reach_target_delta_eef.yaml")
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    observation_provider = build_observation_provider(config, camera_manager)
    observation_provider.configure_sim(sim)
    try:
        try:
            obs = sim.reset(config.task, seed=42)
        except RuntimeError as exc:
            if os.environ.get("MLLM_TELEOP_REQUIRE_RLBENCH") == "1":
                raise
            pytest.skip(f"RLBench/CoppeliaSim runtime unavailable: {exc}")
        assert obs.step_id == 0

        frame = sim.render(config.camera.cameras[0])
        assert np.asarray(frame).shape[-1] == 3
        initial_state = sim.get_state()

        result = sim.step(RLBenchAction(delta_position=(0.0, 0.0, 0.0), gripper=1.0))
        assert result.observation.step_id == 1
        assert result.info["step_id"] == 1

        repeated_obs = sim.reset(config.task, seed=42)
        repeated_state = sim.get_state()
        repeated_frame = sim.render(config.camera.cameras[0])
        assert repeated_obs.step_id == 0
        assert repeated_state["variation_index"] == initial_state["variation_index"]
        assert repeated_state["descriptions"] == initial_state["descriptions"]
        np.testing.assert_allclose(
            repeated_state["task_low_dim_state"],
            initial_state["task_low_dim_state"],
        )
        pixel_delta = np.abs(
            np.asarray(repeated_frame, dtype=np.int16)
            - np.asarray(frame, dtype=np.int16)
        )
        assert float(pixel_delta.mean()) < 1.0
        assert float(np.count_nonzero(pixel_delta) / pixel_delta.size) < 0.01
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_rlbench_fake_client_closes_four_step_public_rgb_loop(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    try:
        __import__("rlbench")
        __import__("pyrep")
    except ImportError as exc:
        pytest.skip(f"RLBench/PyRep are not installed in this environment: {exc}")

    client = RLBenchFakeClient()
    native_actions = []
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-rlbench-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-rlbench.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_RLBENCH_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = RLBenchSimAdapter.get_state
    original_native_action = RLBenchSimAdapter._native_action

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    def recording_native_action(self, action):
        native = original_native_action(self, action)
        native_actions.append(native.copy())
        return native

    monkeypatch.setattr(RLBenchSimAdapter, "get_state", poisoned_get_state)
    monkeypatch.setattr(
        RLBenchSimAdapter,
        "_native_action",
        recording_native_action,
    )
    config = load_config(config_path)
    patched = replace(
        config,
        task=replace(config.task, max_steps=4),
        logging=replace(
            config.logging,
            output_dir=str(tmp_path / mapping_strategy),
            save_video=False,
        ),
    )

    try:
        episode = BenchmarkRunner(patched).run((0,)).episodes[0]
    except RuntimeError as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_RLBENCH") == "1":
            raise
        pytest.skip(f"RLBench/CoppeliaSim runtime unavailable: {exc}")

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_request_image_count(request) for request in client.requests] == [
        1,
        2,
        3,
        4,
    ]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]
    assert len(native_actions) == 4
    assert all(action.shape == (8,) for action in native_actions)
    assert all(float(action[-1]) == 1.0 for action in native_actions)

    latest_request = client.requests[-1]
    assistant_messages = [
        message
        for message in latest_request.messages
        if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 3
    assert all(
        json.loads(message["content"])["Reflection"]
        for message in assistant_messages
    )
    request_text = _request_text(latest_request)
    assert "Move the robot end-effector to the visible target." in request_text
    assert "Camera order: front" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "RLBench world-frame x, y, and z axes" in request_text
    assert "0 means open, 1 means closed" in request_text
    if mapping_strategy == "fixed":
        assert "joystick_0 -> dx" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text

    encoded_requests = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    for sentinel in (*PRIVATE_SENTINELS.values(), "PRIVATE_RLBENCH_API_KEY"):
        assert sentinel not in encoded_requests
    lower_request_text = request_text.lower()
    for forbidden in (
        "gripper_pose",
        "ee_position",
        "tcp_position",
        "task_low_dim_state",
        "joint_positions",
        "variation_index",
        "native action",
        "simulator action",
        "target position",
        "distance",
        "reward",
        "success oracle",
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
    assert all(
        len(step["sim_action"]["delta_position"]) == 3
        and len(step["sim_action"]["delta_rotation_rpy"]) == 3
        and step["sim_action"]["gripper"] == 1.0
        for step in trajectory
    )

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 4
    rendered_frames = []
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (240, 320, 3)
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0
        rendered_frames.append(pixels)
    for previous, current in zip(rendered_frames, rendered_frames[1:]):
        pixel_delta = np.abs(
            current.astype(np.int16) - previous.astype(np.int16)
        )
        assert float(pixel_delta.mean()) < 5.0
        assert float(np.count_nonzero(pixel_delta) / pixel_delta.size) < 0.15

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v2"
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
    if mapping_strategy == "fixed":
        assert "__mapping-" not in episode_dir.parent.parent.name
        assert "__frame-" not in episode_dir.parent.parent.name
    else:
        assert "__mapping-permuted_sign_flipped" in episode_dir.parent.parent.name
        assert "__frame-random_rotation_3d" in episode_dir.parent.parent.name

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            contents = artifact.read_bytes()
            assert b"PRIVATE_RLBENCH_API_KEY" not in contents
    for request_path in episode_dir.glob("model_calls/step_*/request.json"):
        contents = request_path.read_bytes()
        for sentinel in PRIVATE_SENTINELS.values():
            assert sentinel.encode() not in contents


def _request_image_count(request) -> int:
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


def test_colosseum_route_is_blocked_until_rlbench_smoke_is_verified():
    pytest.skip(
        "Colosseum is intentionally not wired yet: first verify RLBench reset/step/render against "
        "CoppeliaSim/PyRep in the rlbench conda backend; see docs/rlbench_colosseum_integration.md."
    )
