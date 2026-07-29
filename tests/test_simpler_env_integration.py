import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.simpler_env import SimplerEnvDeltaEEFController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    control_interface_metadata,
)
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.sim.simpler_env.adapter import SimplerEnvSimAdapter


BASELINE_CONFIG = "configs/simpler_env_google_robot_pick_coke_can_mllm_reflection.yaml"
PERTURBED_CONFIG = "configs/simpler_env_google_robot_pick_coke_can_mllm_reflection_perturbed.yaml"
MLLM_CONFIGS = (
    (BASELINE_CONFIG, "fixed", "identity", True),
    (
        PERTURBED_CONFIG,
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)
PUBLIC_CAMERAS = ("image/base_camera", "image/overhead_camera")
PRIVATE_SENTINELS = {
    "state": "PRIVATE_STATE_SENTINEL",
    "reward": "PRIVATE_REWARD_SENTINEL",
    "success_oracle": "PRIVATE_SUCCESS_SENTINEL",
    "distance": "PRIVATE_DISTANCE_SENTINEL",
    "object_pose": "PRIVATE_OBJECT_POSE_SENTINEL",
    "eef_pose": "PRIVATE_EEF_POSE_SENTINEL",
    "native_sim_action": "PRIVATE_SIM_ACTION_SENTINEL",
    "seed": "PRIVATE_SEED_SENTINEL",
    "permutation": "PRIVATE_PERMUTATION_SENTINEL",
    "signs": "PRIVATE_SIGNS_SENTINEL",
    "rotation_sample": "PRIVATE_ROTATION_SENTINEL",
}


class FakeSimplerEnvClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    **{f"joystick_{index}": 0 for index in range(6)},
                    "joystick_6": 0.0,
                },
                "Reflection": (
                    "The current RGB views do not yet show a reliable motion "
                    "effect, so keep the pose unchanged and the gripper open."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-simpler-env-{len(self.requests)}",
            "model": "fake-simpler-env-model",
            "choices": [{"message": {"content": text}}],
        }
        return ModelResponse(
            text=text,
            request_id=raw["id"],
            model=raw["model"],
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=24,
                cached_input_tokens=20,
            ),
            raw=raw,
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _require_simpler_env() -> None:
    pytest.importorskip("simpler_env")


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_simpler_env_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.logging.save_video is False
    assert config.agent.observation.cameras == PUBLIC_CAMERAS
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

    control = raw["control_interface"]
    assert control["control_frame_description"] == (
        "the Google Robot end-effector origin with axes aligned to the robot base frame"
    )
    assert control["gripper_open_command"] == 0
    assert control["gripper_closed_command"] == 1
    assert control["params"]["rotation_representation"] == "axis_angle"

    group = make_experiment_group(config)
    assert group is not None
    assert "output-action_with_reflection" in group
    if mapping_strategy == "fixed":
        assert "__mapping-" not in group
        assert "__frame-" not in group
    else:
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_simpler_env_mllm_task_text_is_identical_between_control_modes():
    baseline = load_config(BASELINE_CONFIG)
    perturbed = load_config(PERTURBED_CONFIG)

    assert baseline.task.params["instruction"] == perturbed.task.params["instruction"]
    assert baseline.task.params["instruction"] == "pick coke can"
    assert "joystick" not in baseline.task.params["instruction"].lower()
    assert "rotation" not in baseline.task.params["instruction"].lower()


def test_simpler_env_public_prompt_describes_controls_without_hidden_samples():
    baseline = load_config(BASELINE_CONFIG)
    baseline_prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text
    perturbed = load_config(PERTURBED_CONFIG)
    perturbed_metadata = control_interface_metadata(
        perturbed,
        perturbed.benchmark.seed,
    )
    perturbed_prompt = ControlInstructionBuilder().build(
        perturbed_metadata,
        build_action_space(perturbed).schema(),
    ).text

    assert "dx, dy, and dz denote end-effector translation" in baseline_prompt
    assert "droll, dpitch, and dyaw denote end-effector rotation" in baseline_prompt
    assert "joystick_0 -> dx" in baseline_prompt
    assert "joystick_5 -> dyaw" in baseline_prompt
    assert "direct control" in baseline_prompt
    assert "final virtual joystick" in baseline_prompt
    assert "No additional control-frame rotation is applied" in baseline_prompt

    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert (
        "Infer the hidden controls from visual observations and your own action "
        "history"
    ) in perturbed_prompt
    assert "joystick_0 ->" not in perturbed_prompt
    assert "mapping" not in perturbed_metadata
    assert "mapping_strategy" not in perturbed_metadata
    assert "frame_transform" not in perturbed_metadata


def test_simpler_env_perturbation_keeps_gripper_unmodified():
    config = load_config(PERTURBED_CONFIG)
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller, SimplerEnvDeltaEEFController)
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

    action = JoystickAction(
        {
            **{f"joystick_{index}": 0.0 for index in range(6)},
            "joystick_6": 1.0,
        }
    )
    sim_action = controller.translate(action)
    assert np.asarray(sim_action.values).shape == (7,)
    assert sim_action.values[-1] == 1.0
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }


def test_simpler_env_adapter_serializes_pose_like_info():
    config = load_config(BASELINE_CONFIG)
    adapter = SimplerEnvSimAdapter(config.simulator)
    pose = type(
        "PoseLike",
        (),
        {
            "p": np.asarray([0.1, 0.2, 0.3]),
            "q": np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
    )()

    assert adapter._jsonable({"tcp_pose": pose}) == {
        "tcp_pose": {
            "position": [0.1, 0.2, 0.3],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    }


def test_simpler_env_runtime_rgb_and_native_action_contract():
    _require_simpler_env()
    config = load_config(BASELINE_CONFIG)
    sim = build_sim(config)
    provider = build_observation_provider(
        config,
        build_camera_manager(config.camera, config.benchmark.seed),
    )
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )

    try:
        sim.reset(config.task, seed=config.benchmark.seed)
        observation = provider.get_observation(sim)
        assert set(PUBLIC_CAMERAS).issubset(observation.images)
        for camera in PUBLIC_CAMERAS:
            image = np.asarray(observation.images[camera])
            assert image.ndim == 3
            assert image.shape[-1] == 3
            assert image.dtype == np.uint8
            assert float(image.mean()) > 1.0
            assert float(image.std()) > 1.0

        action = JoystickAction(
            {
                **{f"joystick_{index}": 0.0 for index in range(6)},
                "joystick_6": 0.0,
            }
        )
        sim_action = controller.translate(action)
        native_action, clip_metadata = sim._native_action(sim_action)
        assert native_action.shape == (7,)
        np.testing.assert_array_less(
            np.asarray(sim.env.action_space.low) - 1e-7,
            native_action,
        )
        np.testing.assert_array_less(
            native_action,
            np.asarray(sim.env.action_space.high) + 1e-7,
        )
        assert clip_metadata["clipped"] is False

        step = sim.step(sim_action)
        assert step.observation.step_id == 1
        next_observation = provider.get_observation(sim)
        assert set(PUBLIC_CAMERAS).issubset(next_observation.images)
        assert all(
            float(np.asarray(next_observation.images[camera]).std()) > 1.0
            for camera in PUBLIC_CAMERAS
        )
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_simpler_env_fake_client_closes_public_rgb_loop_without_private_state(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    _require_simpler_env()
    client = FakeSimplerEnvClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-simpler-env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_API_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = SimplerEnvSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    monkeypatch.setattr(SimplerEnvSimAdapter, "get_state", poisoned_get_state)
    config = load_config(config_path)
    config = replace(
        config,
        task=replace(config.task, max_steps=4, success_threshold=2.0),
        logging=replace(
            config.logging,
            output_dir=str(tmp_path / mapping_strategy),
            save_video=False,
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_request_image_count(request) for request in client.requests] == [
        2,
        4,
        6,
        8,
    ]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    request_text = "\n".join(
        _request_text(request)
        for request in client.requests
    )
    assert "Camera order: image/base_camera, image/overhead_camera" in request_text
    assert "Segmentation" not in request_text
    assert "pick coke can" in request_text
    assert "success" not in request_text.lower()
    for forbidden in (
        *PRIVATE_SENTINELS.values(),
        "PRIVATE_API_KEY_SENTINEL",
        "permuted_sign_flipped",
        "random_rotation_3d",
        "rotation_matrix",
        "quaternion_xyzw",
    ):
        assert forbidden not in request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)
    assert all(len(step["sim_action"]["values"]) == 7 for step in trajectory)
    assert all(
        np.all(np.isfinite(step["sim_action"]["values"]))
        for step in trajectory
    )

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 8
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.ndim == 3
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v2"
    assert manifest["history"]["action_window"] == "all"
    assert manifest["history"]["image_window"] == 3
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert manifest["control_interface"]["mapping_visible_to_agent"] is visible
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            contents = artifact.read_bytes()
            assert b"PRIVATE_API_KEY_SENTINEL" not in contents
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
