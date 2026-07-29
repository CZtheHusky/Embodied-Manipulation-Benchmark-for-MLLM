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
    ("config_path", "mapping_strategy", "frame_transform"),
    [
        (
            "configs/pusher_delta_eef_mllm_reflection.yaml",
            "fixed",
            "identity",
        ),
        (
            "configs/pusher_delta_eef_mllm_reflection_perturbed.yaml",
            "permuted_sign_flipped",
            "random_rotation_3d",
        ),
    ],
)
def test_pusher_reflection_configs_are_explicit_and_distinct(
    config_path,
    mapping_strategy,
    frame_transform,
):
    config = load_config(config_path)

    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    group = make_experiment_group(config)
    assert group is not None
    assert "output-action_with_reflection" in group
    if mapping_strategy != "fixed":
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_pusher_fake_client_closes_public_visual_loop(
    monkeypatch,
    tmp_path,
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
    config = load_config(
        "configs/pusher_delta_eef_mllm_reflection.yaml",
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
    request_text = client.requests[-1].encoded_body().decode("utf-8")
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "Gymnasium Pusher world x, y, and z axes" in request_text
    assert "Move the white cylinder to the red goal disk" in request_text
    for forbidden in (
        "object_goal_distance",
        "fingertip_object_distance",
        "object_position",
        "goal_position",
        "fingertip_position",
        "ee_target_xyz",
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
    pngs = sorted(episode_dir.glob("model_calls/images/step_*/front.png"))
    assert len(pngs) == 2
    from PIL import Image

    assert all(np.asarray(Image.open(path).convert("RGB")).max() > 0 for path in pngs)
