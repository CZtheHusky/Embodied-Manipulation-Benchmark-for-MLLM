import json
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.agents.model_client import (
    ModelClientError,
    ModelResponse,
    ModelUsage,
)
from mllm_teleop_benchmark.config.schema import (
    AgentConfig,
    AgentHistoryConfig,
    AgentObservationConfig,
    ModelConfig,
)
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group


def _valid_action_text():
    channels = ",".join(f'"joystick_{index}":0' for index in range(7))
    return '{"channels":{' + channels + "}}"


class FakeClient:
    def __init__(self, response_text=None, error=False):
        self.response_text = response_text or _valid_action_text()
        self.error = error
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self.error:
            raise ModelClientError(
                "endpoint unavailable",
                attempts=3,
                status_code=503,
                latency_ms=12.0,
                request_bytes=len(request.encoded_body()),
            )
        raw = {
            "id": f"fake-{len(self.requests)}",
            "model": "fake-model",
            "choices": [{"message": {"content": self.response_text}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        }
        return ModelResponse(
            text=self.response_text,
            request_id=raw["id"],
            model=raw["model"],
            usage=ModelUsage(input_tokens=100, output_tokens=8, cached_input_tokens=20),
            raw=raw,
            latency_ms=5.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _config(tmp_path, image_window, action_output_mode="action_only"):
    config = load_config("configs/dummy_reach.yaml")
    agent = AgentConfig(
        type="mllm",
        action_output_mode=action_output_mode,
        invalid_action_recovery="noop",
        model=ModelConfig(
            name="fake-model",
            base_url="http://fake-model.test/v1",
            api_key_env="TEST_MLLM_API_KEY",
            require_api_key=True,
        ),
        observation=AgentObservationConfig(cameras=("front",), image_max_side=32),
        history=AgentHistoryConfig(
            action_window="all",
            image_window=image_window,
            max_images_per_request=128,
            max_request_bytes=67_108_864,
        ),
    )
    return replace(
        config,
        benchmark=replace(config.benchmark, num_episodes=1),
        task=replace(config.task, max_steps=3),
        agent=agent,
        logging=replace(config.logging, output_dir=tmp_path),
    )


def _image_count(request):
    return sum(
        1
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


def _run_with_fake(monkeypatch, config, client):
    monkeypatch.setattr(
        "mllm_teleop_benchmark.sim.dummy.DummySimAdapter.render",
        lambda self, _camera: np.full((16, 16, 3), self.step_id, dtype=np.uint8),
    )
    monkeypatch.setenv("TEST_MLLM_API_KEY", "VERY_PRIVATE_API_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )
    return BenchmarkRunner(config).run().episodes[0]


@pytest.mark.parametrize(
    ("image_window", "expected_counts"),
    [
        (0, [1, 1, 1]),
        (2, [1, 2, 3]),
        ("all", [1, 2, 3]),
    ],
)
def test_fake_client_runs_three_step_episode_with_artifacts_and_history(
    monkeypatch,
    tmp_path,
    image_window,
    expected_counts,
):
    client = FakeClient()
    episode = _run_with_fake(monkeypatch, _config(tmp_path, image_window), client)

    assert episode.steps == 3
    assert episode.termination_reason == "max_steps"
    assert [_image_count(request) for request in client.requests] == expected_counts
    assert episode.metrics["model_call_count"] == 3
    assert episode.metrics["model_api_error_count"] == 0
    assert episode.metrics["cached_input_tokens"] == 60
    assert episode.metrics["cache_hit_ratio"] == pytest.approx(0.2)
    assert episode.metrics["cache_metrics_status"] == "reported"

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trajectory) == 3
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["agent_trace"]["actor_type"] == "mllm" for step in trajectory)
    assert (episode_dir / "model_calls/step_0000/request.json").exists()
    assert (episode_dir / "model_calls/step_0002/response.json").exists()
    assert (episode_dir / "model_calls/images/step_0000/front.png").exists()

    request_artifact = (episode_dir / "model_calls/step_0002/request.json").read_text(encoding="utf-8")
    assert "data:image/png;base64" not in request_artifact
    assert "data:image/png;sha256=" in request_artifact
    run_dir = episode_dir.parent
    group_dir = run_dir.parent
    assert (
        group_dir.name
        == f"model-fake-model__reasoning-default__api-chat_completions"
        f"__history-images-{image_window}-actions-all__episodes-1__maxsteps-3"
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["requested_model"] == "fake-model"
    assert manifest["model"]["reasoning_effort"] is None
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["history"]["image_window"] == image_window
    assert manifest["control_interface"]["mapping_strategy"] == "fixed"
    assert manifest["control_interface"]["frame_transform"] == "identity"
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["observed_response_models"] == ["fake-model"]
    assert run_summary["aggregate_metrics"]["num_episodes"] == 1
    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"VERY_PRIVATE_API_KEY_SENTINEL" not in artifact.read_bytes()


def test_experiment_group_identifies_control_perturbations(tmp_path):
    config = _config(tmp_path, "all")
    config = replace(
        config,
        control_interface=replace(
            config.control_interface,
            mapping_strategy="permuted_sign_flipped",
            frame_transform="random_rotation_2d",
        ),
    )

    assert make_experiment_group(config) == (
        "model-fake-model__reasoning-default__api-chat_completions"
        "__history-images-all-actions-all__episodes-1__maxsteps-3"
        "__mapping-permuted_sign_flipped__frame-random_rotation_2d"
    )


def test_reflection_mode_records_and_replays_reflection_with_distinct_group(
    monkeypatch,
    tmp_path,
):
    response = (
        _valid_action_text()[:-1]
        + ',"Reflection":"joystick_0 may control horizontal motion"}'
    )
    client = FakeClient(response_text=response)
    episode = _run_with_fake(
        monkeypatch,
        _config(
            tmp_path,
            3,
            action_output_mode="action_with_reflection",
        ),
        client,
    )

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all(
        step["reflection"] == "joystick_0 may control horizontal motion"
        for step in trajectory
    )
    assistant_payload = json.loads(
        next(
            message["content"]
            for message in client.requests[1].messages
            if message["role"] == "assistant"
        )
    )
    assert assistant_payload["Reflection"] == (
        "joystick_0 may control horizontal motion"
    )
    run_dir = episode_dir.parent
    assert run_dir.parent.name.endswith("__output-action_with_reflection")
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v1"


def test_api_failure_terminates_without_stepping_environment(monkeypatch, tmp_path):
    client = FakeClient(error=True)
    monkeypatch.setattr(
        "mllm_teleop_benchmark.sim.dummy.DummySimAdapter.step",
        lambda *_args, **_kwargs: pytest.fail("API errors must not call sim.step"),
    )
    episode = _run_with_fake(monkeypatch, _config(tmp_path, "all"), client)

    assert episode.steps == 0
    assert episode.termination_reason == "agent_error"
    assert len(client.requests) == 1
    assert episode.metrics["invalid_action_count"] == 0
    assert episode.metrics["model_api_error_count"] == 1
    assert episode.metrics["model_retry_count"] == 2
    summary = json.loads((Path(episode.log_dir) / "summary.json").read_text(encoding="utf-8"))
    assert summary["termination_reason"] == "agent_error"


def test_invalid_model_json_executes_noop_without_extra_api_request(monkeypatch, tmp_path):
    client = FakeClient(response_text="{not-json")
    episode = _run_with_fake(monkeypatch, _config(tmp_path, 0), client)

    assert episode.steps == 3
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 3
    assert episode.metrics["invalid_action_count"] == 3
    assert episode.metrics["model_api_error_count"] == 0
