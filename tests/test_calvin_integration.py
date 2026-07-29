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
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import CalvinAction
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


def test_calvin_config_load():
    """Verify that the CALVIN config can be loaded even if calvin-env is missing."""
    config = load_config("configs/calvin_delta_eef.yaml")
    assert config.simulator.type == "calvin"
    assert config.task.type == "calvin_generic"
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

    assert config.task.params["instruction"] == (
        "Pick up the object and place it in the drawer."
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
    assert config.control_interface.control_frame_description == (
        "the CALVIN Cartesian-relative controller's PyBullet world x, y, and z axes"
    )
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


def test_calvin_mllm_prompt_describes_public_pose_and_gripper_contract():
    baseline = load_config("configs/calvin_delta_eef_mllm_reflection.yaml")
    prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text

    assert (
        "active controllable dimensions are: dx, dy, dz, droll, dpitch, dyaw"
        in prompt
    )
    assert "dx, dy, and dz denote end-effector translation" in prompt
    assert "droll, dpitch, and dyaw denote end-effector rotation" in prompt
    assert "positive mapped value commands the positive direction" in prompt
    assert "CALVIN Cartesian-relative controller's PyBullet world" in prompt
    assert "joystick_0 -> dx" in prompt
    assert "joystick_5 -> dyaw" in prompt
    assert "joystick_6 -> gripper" in prompt
    assert "1 means open, 0 means closed" in prompt
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
    assert "joystick_0 ->" not in prompt
    assert "mapping" not in metadata
    assert "mapping_strategy" not in metadata
    assert "frame_transform" not in metadata
    assert "rotation_matrix" not in str(metadata)


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
    assert "Prompt version: mllm_joystick_reflection_v2" in request_text
    assert "Pick up the object and place it in the drawer." in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "droll, dpitch, and dyaw denote end-effector rotation" in request_text
    assert "CALVIN Cartesian-relative controller's PyBullet world" in request_text
    assert "1 means open, 0 means closed" in request_text
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
        assert np.all(np.abs(values[:6]) <= 0.100_001)
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
