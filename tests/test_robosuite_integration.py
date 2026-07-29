import json
import os
from pathlib import Path

import numpy as np
import pytest

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.robosuite_controller import RobosuiteOSCController
from mllm_teleop_benchmark.core.registry import build_action_space, build_controller, build_mapping
from mllm_teleop_benchmark.core.types import RobosuiteAction
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import (
    IdentityControlFrameTransform,
    RotationControlFrameTransform,
)
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.robosuite_adapter import RobosuiteSimAdapter, sample_camera_randomization


MLLM_CONFIGS = (
    (
        "configs/robosuite_lift_mllm_reflection.yaml",
        "fixed",
        "identity",
    ),
    (
        "configs/robosuite_lift_mllm_reflection_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ),
)

PRIVATE_SENTINELS = {
    "robot_proprio": "PRIVATE_STATE_SENTINEL",
    "eef_pos": "PRIVATE_EEF_POSE_SENTINEL",
    "eef_quat": "PRIVATE_EEF_QUAT_SENTINEL",
    "reward": "PRIVATE_REWARD_SENTINEL",
    "object_pose": "PRIVATE_OBJECT_POSE_SENTINEL",
    "distance": "PRIVATE_DISTANCE_SENTINEL",
    "sim_action": "PRIVATE_SIM_ACTION_SENTINEL",
    "seed": "PRIVATE_SEED_SENTINEL",
    "camera_randomization": "PRIVATE_CAMERA_RANDOMIZATION_SENTINEL",
}


class RobosuiteFakeClient:
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
                "Reflection": "More RGB feedback is needed to identify each motion channel.",
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-robosuite-{len(self.requests)}",
            "model": "fake-robosuite-model",
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


def _require_robosuite() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        __import__("robosuite")
    except ImportError as exc:
        pytest.skip(f"robosuite is not available in this environment: {exc}")
    except Exception as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_ROBOSUITE") == "1" or "robosuite" in os.environ.get("CONDA_DEFAULT_ENV", ""):
            raise
        pytest.skip(f"robosuite failed to import in this optional-backend environment: {exc}")


def test_robosuite_config_uses_worker_play_runtime():
    config = load_config("configs/robosuite_lift.yaml")

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "agentview"


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform"),
    MLLM_CONFIGS,
)
def test_robosuite_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
):
    config = load_config(config_path)
    action_schema = build_action_space(config).schema()

    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.response_format == "json_object"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == (
        "agentview",
        "robot0_eye_in_hand",
    )
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.gripper == "direct"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert tuple(action_schema) == (
        "joystick_0",
        "joystick_1",
        "joystick_2",
        "joystick_3",
    )
    assert all(
        action_schema[f"joystick_{index}"]
        == {"kind": "discrete", "values": [-1.0, 0.0, 1.0]}
        for index in range(3)
    )
    assert action_schema["joystick_3"] == {
        "kind": "continuous",
        "min": 0.0,
        "max": 1.0,
    }

    group = make_experiment_group(config)
    assert group is not None
    assert "output-action_with_reflection" in group
    if mapping_strategy != "fixed":
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_robosuite_perturbation_only_changes_motion_channels():
    config = load_config("configs/robosuite_lift_mllm_reflection_perturbed.yaml")
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller.frame_transform, RotationControlFrameTransform)
    assert set(mapping.channels.values()) == {"dx", "dy", "dz", "gripper"}
    assert mapping.channels["joystick_3"] == "gripper"
    assert mapping.signs["joystick_3"] == 1.0
    assert any(
        mapping.signs[f"joystick_{index}"] < 0.0
        for index in range(3)
    )

    action = JoystickAction(
        {
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": 0.0,
            "joystick_3": 1.0,
        }
    )
    sim_action = controller.translate(action)
    assert len(sim_action.values) == 7
    assert sim_action.values[6] == 1.0
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }


def test_robosuite_adapter_init():
    _require_robosuite()
    config = load_config("configs/robosuite_lift.yaml")
    adapter = RobosuiteSimAdapter(config.simulator)
    try:
        obs = adapter.reset(config.task, seed=42)

        assert "agentview" in obs.images
        assert "robot0_eye_in_hand" in obs.images
        assert obs.images["agentview"].shape == (240, 320, 3)
        assert all(
            image.dtype == np.uint8 and int(image.max()) > int(image.min())
            for image in obs.images.values()
        )

        state = adapter.get_state()
        assert state["step_id"] == 0
        assert "eef_pos" in state
        assert "is_success" in state

        step = adapter.step(RobosuiteAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)))
        assert step.observation.step_id == 1
        assert adapter.get_state()["step_id"] == 1

        rendered = adapter.render(config.camera.cameras[0])
        assert rendered.shape == (240, 320, 3)
        assert rendered.dtype == np.uint8
        assert int(rendered.max()) > int(rendered.min())
    finally:
        adapter.close()


def test_robosuite_camera_randomization_sampling_is_seeded():
    cfg = {
        "agentview": {
            "pos_range": [[0.9, 1.1], [-0.1, 0.1], [1.4, 1.6]],
            "fovy_range": [40.0, 50.0],
        }
    }

    first = sample_camera_randomization(cfg, seed=5)
    second = sample_camera_randomization(cfg, seed=5)
    third = sample_camera_randomization(cfg, seed=6)

    assert first == second
    assert first != third
    assert len(first["agentview"]["pos"]) == 3
    assert 40.0 <= first["agentview"]["fovy"] <= 50.0


def test_robosuite_controller_translation():
    mapping = FixedJoystickMapping(
        channels={"joystick_0": "dx", "joystick_1": "dy", "joystick_2": "dz", "joystick_3": "gripper"},
        signs={"joystick_0": 1.0, "joystick_1": 1.0, "joystick_2": 1.0, "joystick_3": 1.0}
    )
    transform = IdentityControlFrameTransform()
    scales = {"dx": 0.1, "dy": 0.1, "dz": 0.1, "gripper": 1.0}
    controller = RobosuiteOSCController(mapping, transform, scales)
    
    # Test movement - include all mapped channels
    action = JoystickAction(channels={"joystick_0": 1.0, "joystick_1": -1.0, "joystick_2": 0.5, "joystick_3": 0.0})
    sim_action = controller.translate(action)
    
    assert isinstance(sim_action, RobosuiteAction)
    values = np.asarray(sim_action.values)
    assert values.shape == (7,)
    # dx = 1.0 * 0.1 = 0.1
    # dy = -1.0 * 0.1 = -0.1
    # dz = 0.5 * 0.1 = 0.05
    np.testing.assert_allclose(values[:3], [0.1, -0.1, 0.05])
    
    # Test gripper: benchmark 0.0 -> robosuite -1.0
    action_open = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 0.0})
    sim_action_open = controller.translate(action_open)
    assert sim_action_open.values[6] == -1.0
    
    # Test gripper: benchmark 1.0 -> robosuite 1.0
    action_close = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 1.0})
    sim_action_close = controller.translate(action_close)
    assert sim_action_close.values[6] == 1.0


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "_frame_transform"),
    MLLM_CONFIGS,
)
def test_robosuite_fake_client_closes_public_rgb_loop_without_private_state(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    _frame_transform,
):
    _require_robosuite()
    client = RobosuiteFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-robosuite-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_API_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = RobosuiteSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    monkeypatch.setattr(RobosuiteSimAdapter, "get_state", poisoned_get_state)
    config = load_config(
        config_path,
        (
            "task.max_steps=2",
            f"logging.output_dir={tmp_path}",
            "logging.save_video=false",
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 2
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 2
    assert [_image_count(request) for request in client.requests] == [2, 4]
    assert "Observation step 0." in _request_text(client.requests[0])
    assert "Observation step 1." in _request_text(client.requests[1])

    assistant_messages = [
        message
        for message in client.requests[1].messages
        if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1
    prior_action = json.loads(assistant_messages[0]["content"])
    assert set(prior_action["channels"]) == {
        "joystick_0",
        "joystick_1",
        "joystick_2",
        "joystick_3",
    }
    assert prior_action["Reflection"]

    request_text = _request_text(client.requests[-1])
    assert "active controllable dimensions are: dx, dy, dz" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert '"joystick_3":{"kind":"continuous","max":1.0,"min":0.0}' in request_text
    assert "direct control" in request_text
    assert "final virtual joystick" in request_text
    assert "0 means open, 1 means closed, 0.5 means hold" in request_text
    assert "robosuite OSC world x, y, and z axes" in request_text
    if mapping_strategy == "fixed":
        assert "joystick_0 -> dx" in request_text
        assert "joystick_1 -> dy" in request_text
        assert "joystick_2 -> dz" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert (
            "hidden joystick mapping and control-frame transform remain fixed "
            "for the entire episode"
        ) in request_text

    for forbidden in (
        *PRIVATE_SENTINELS.values(),
        "PRIVATE_API_KEY_SENTINEL",
        "robot0_proprio-state",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "object_pose",
        "sim_action",
        "camera_randomization",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in request_text
    lower_request_text = request_text.lower()
    for forbidden_term in (
        "simulator state",
        "reward",
        "object pose",
        "eef pose",
        "end-effector pose",
        "distance",
        "sim action",
        "simulator action",
        "episode seed",
    ):
        assert forbidden_term not in lower_request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(trajectory) == 2
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 4
    from PIL import Image

    assert all(
        int(np.asarray(Image.open(path).convert("RGB")).max()) > 0
        for path in pngs
    )


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
