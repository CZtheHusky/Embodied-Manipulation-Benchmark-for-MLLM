from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.agents.model_client import (
    ModelResponse,
    ModelUsage,
)
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group


PUSHER_MATRIX = (
    (
        "configs/pusher_delta_eef_mllm_chat_completions.yaml",
        "chat_completions",
        "mujoco_randomized",
        "random_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/pusher_delta_eef_mllm_chat_completions_perturbed.yaml",
        "chat_completions",
        "mujoco_randomized",
        "random_view",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
    (
        "configs/pusher_delta_eef_mllm_chat_completions_default_view.yaml",
        "chat_completions",
        "fixed",
        "default_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/pusher_delta_eef_mllm_chat_completions_default_view_perturbed.yaml",
        "chat_completions",
        "fixed",
        "default_view",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
    (
        "configs/pusher_delta_eef_mllm_responses.yaml",
        "responses",
        "mujoco_randomized",
        "random_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/pusher_delta_eef_mllm_responses_perturbed.yaml",
        "responses",
        "mujoco_randomized",
        "random_view",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
    (
        "configs/pusher_delta_eef_mllm_responses_default_view.yaml",
        "responses",
        "fixed",
        "default_view",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/pusher_delta_eef_mllm_responses_default_view_perturbed.yaml",
        "responses",
        "fixed",
        "default_view",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)


class PusherFakeClient:
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
                },
                "Reflection": (
                    "The three joysticks control end-effector translation axes; "
                    "more visual feedback is needed to determine their effects."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-pusher-{len(self.requests)}",
            "model": "fake-pusher-model",
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


@pytest.mark.parametrize(
    (
        "config_path",
        "api",
        "camera_manager",
        "camera_condition",
        "mapping_strategy",
        "frame_transform",
        "visible",
    ),
    PUSHER_MATRIX,
)
def test_pusher_matrix_configs_are_explicit_and_distinct(
    config_path,
    api,
    camera_manager,
    camera_condition,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)

    assert config.benchmark.num_episodes == 10
    assert config.agent.model is not None
    assert config.agent.model.api == api
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == (
        "all" if api == "responses" else 3
    )
    assert config.logging.save_video is True
    assert config.camera.manager == camera_manager
    assert config.camera.params["experiment_condition"] == camera_condition
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible

    description = config.control_interface.control_frame_description
    assert description is not None
    if camera_condition == "random_view":
        assert "randomly sampled at episode reset" in description
        assert "fixed for that episode" in description
        assert "do not have a fixed world-axis projection across episodes" in description
    else:
        assert "environment's default pose" in description
        assert "remains fixed across episodes and steps" in description
        assert "consistent world-axis projection" in description

    group = make_experiment_group(config)
    assert group is not None
    assert f"api-{api}" in group
    assert f"camera-{camera_condition}" in group
    assert "__output-" not in group
    if mapping_strategy != "fixed":
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


@pytest.mark.parametrize(
    (
        "config_path",
        "api",
        "camera_manager",
        "camera_condition",
        "mapping_strategy",
        "frame_transform",
        "visible",
    ),
    PUSHER_MATRIX,
)
def test_pusher_fake_client_closes_privilege_safe_visual_loop(
    monkeypatch,
    tmp_path,
    config_path,
    api,
    camera_manager,
    camera_condition,
    mapping_strategy,
    frame_transform,
    visible,
):
    pytest.importorskip("gymnasium")
    client = PusherFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-pusher-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAIResponsesClient",
        lambda **_kwargs: client,
    )
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
    request_text = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    assert "You may include the top-level string key Reflection" in request_text
    assert "what each joystick channel controls" in request_text
    assert "Each motion joystick accepts any finite numeric value" in request_text
    assert "Values outside [-30, 30] are clipped" in request_text
    assert "A value of 1 requests one configured default motion step" in request_text
    assert "increase that motion value's magnitude" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "Gymnasium Pusher world x, y, and z axes" in request_text
    assert "moves the end-effector target by 0.01 meters" in request_text
    assert "Move the white cylinder to the red goal disk" in request_text
    if camera_condition == "random_view":
        assert "front camera pose is randomly sampled at episode reset" in request_text
        assert "remains fixed for that episode" in request_text
        assert "do not have a fixed world-axis projection across episodes" in request_text
    else:
        assert "front camera uses the environment's default pose" in request_text
        assert "remains fixed across episodes and steps" in request_text
        assert "consistent world-axis projection" in request_text
    if visible:
        assert "joystick_0 -> dx" in request_text
        assert "joystick_1 -> dy" in request_text
        assert "joystick_2 -> dz" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert "permuted_sign_flipped" not in request_text
        assert "random_rotation_3d" not in request_text
        assert "rotation_matrix" not in request_text
    for forbidden in (
        "object_goal_distance",
        "fingertip_object_distance",
        "object_position",
        "goal_position",
        "goal_hold_counter",
        "fingertip_position",
        "ee_target_xyz",
        "ee_target_reachability_clamped",
        "last_joint_position_target",
        "mujoco_camera_config",
        "distance_range",
        "azimuth_range",
        "elevation_range",
        "PRIVATE_KEY_SENTINEL",
    ):
        assert forbidden not in request_text

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
    assert all(
        set(step["parsed_action"])
        == {"joystick_0", "joystick_1", "joystick_2"}
        for step in trajectory
    )
    assert all(step["agent_trace"]["api"] == api for step in trajectory)
    pngs = sorted(episode_dir.glob("model_calls/images/step_*/front.png"))
    assert len(pngs) == 2
    from PIL import Image

    assert all(np.asarray(Image.open(path).convert("RGB")).max() > 0 for path in pngs)

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["camera"] == {
        "experiment_condition": camera_condition,
        "manager": camera_manager,
    }
