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
from mllm_teleop_benchmark.sim.dummy import DummySimAdapter


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


class SequencedFakeClient(FakeClient):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def complete(self, request):
        self.response_text = self.responses.pop(0)
        return super().complete(request)


def _config(tmp_path, image_window):
    config = load_config("configs/dummy_reach.yaml")
    control_interface = replace(
        config.control_interface,
        channel_kind="continuous",
        channel_values=(),
        channel_minimum=-30.0,
        channel_maximum=30.0,
    )
    agent = AgentConfig(
        type="mllm",
        invalid_action_recovery="noop",
        model=ModelConfig(
            name="fake-model",
            base_url="http://fake-model.test/v1",
            api_key_env="TEST_MLLM_API_KEY",
            require_api_key=True,
        ),
        observation=AgentObservationConfig(cameras=("front",), image_max_side=32),
        history=AgentHistoryConfig(
            action_window=image_window,
            image_window=image_window,
            max_images_per_request=128,
            max_request_bytes=67_108_864,
        ),
    )
    return replace(
        config,
        benchmark=replace(config.benchmark, num_episodes=1),
        task=replace(config.task, max_steps=3),
        action_space=control_interface.to_action_space_config(),
        control_interface=control_interface,
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


def _capture_dummy_actions(monkeypatch):
    actions = []
    original_step = DummySimAdapter.step

    def capture_step(sim, action):
        actions.append(action)
        return original_step(sim, action)

    monkeypatch.setattr(DummySimAdapter, "step", capture_step)
    return actions


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
    assert episode.metrics["agent_steps"] == 3
    assert episode.metrics["simulator_substeps"] == 3
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
        f"__prompt-mllm_joystick_chat_completions_v11"
        f"__history-images-{image_window}-actions-{image_window}__episodes-1__maxsteps-3"
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["step_budget_unit"] == "agent_command"
    assert manifest["max_agent_steps"] == 3
    assert manifest["max_internal_substeps_per_agent_step"] == 30
    assert manifest["required_simulator_steps"] == 90
    assert manifest["configured_simulator_step_horizon"] == 90
    assert manifest["source"]["status"] == "reported"
    assert len(manifest["source"]["git_commit"]) == 40
    assert isinstance(manifest["source"]["git_dirty"], bool)
    assert manifest["source"]["config_path"] == "configs/dummy_reach.yaml"
    assert manifest["simulator"]["type"] == "dummy"
    assert manifest["model"]["requested_model"] == "fake-model"
    assert manifest["model"]["reasoning_effort"] is None
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["history"]["image_window"] == image_window
    assert manifest["control_interface"]["mapping_strategy"] == "fixed"
    assert manifest["control_interface"]["frame_transform"] == "identity"
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["observed_response_models"] == ["fake-model"]
    assert run_summary["aggregate_metrics"]["num_episodes"] == 1
    assert run_summary["step_budget_unit"] == "agent_command"
    assert run_summary["episodes"][0]["agent_steps"] == 3
    assert run_summary["episodes"][0]["simulator_substeps"] == 3
    summary = json.loads(
        (episode_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["step_budget_unit"] == "agent_command"
    assert summary["max_agent_steps"] == 3
    assert summary["agent_steps"] == 3
    assert summary["simulator_substeps"] == 3
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
        "__prompt-mllm_joystick_chat_completions_v11"
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
    assert "__prompt-mllm_joystick_chat_completions_v11__" in run_dir.parent.name
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert "action_output_mode" not in manifest["model"]
    assert manifest["model"]["prompt_version"] == "mllm_joystick_chat_completions_v11"


def test_responses_experiment_group_records_prompt_version(tmp_path):
    config = _config(tmp_path, "all")
    config = replace(
        config,
        agent=replace(
            config.agent,
            model=replace(config.agent.model, api="responses"),
        ),
    )

    group = make_experiment_group(config)

    assert group is not None
    assert "__api-responses__prompt-mllm_joystick_responses_v10__history-full__" in group


def test_api_failure_terminates_without_stepping_environment(monkeypatch, tmp_path):
    client = FakeClient(error=True)
    monkeypatch.setattr(
        "mllm_teleop_benchmark.sim.dummy.DummySimAdapter.step",
        lambda *_args, **_kwargs: pytest.fail("API errors must not call sim.step"),
    )
    episode = _run_with_fake(monkeypatch, _config(tmp_path, "all"), client)

    assert episode.steps == 0
    assert episode.termination_reason == "agent_error"
    assert episode.metrics["agent_steps"] == 0
    assert episode.metrics["simulator_substeps"] == 0
    assert len(client.requests) == 1
    assert episode.metrics["invalid_action_count"] == 0
    assert episode.metrics["model_api_error_count"] == 1
    assert episode.metrics["model_retry_count"] == 2
    summary = json.loads((Path(episode.log_dir) / "summary.json").read_text(encoding="utf-8"))
    assert summary["termination_reason"] == "agent_error"


def test_http_200_without_final_content_persists_sanitized_response(
    monkeypatch,
    tmp_path,
):
    class EmptyFinalClient(FakeClient):
        def complete(self, request):
            self.requests.append(request)
            raise ModelClientError(
                "Model response message does not contain text content",
                attempts=1,
                status_code=200,
                raw_response={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": None,
                                "reasoning_content": "private chain of thought",
                            },
                        }
                    ]
                },
            )

    episode = _run_with_fake(
        monkeypatch,
        _config(tmp_path, "all"),
        EmptyFinalClient(),
    )

    response = json.loads(
        (
            Path(episode.log_dir)
            / "model_calls/step_0000/response.json"
        ).read_text(encoding="utf-8")
    )
    assert response["choices"][0]["finish_reason"] == "length"
    assert (
        response["choices"][0]["message"]["reasoning_content"]
        == "[REDACTED: hidden reasoning]"
    )


def test_invalid_model_json_executes_noop_without_extra_api_request(monkeypatch, tmp_path):
    client = FakeClient(response_text="{not-json")
    config = _config(tmp_path, 0)
    config = replace(
        config,
        control_interface=replace(
            config.control_interface,
            gripper_open_command=0.0,
            gripper_closed_command=1.0,
            gripper_hold_command=0.5,
        ),
    )
    sim_actions = _capture_dummy_actions(monkeypatch)
    episode = _run_with_fake(monkeypatch, config, client)

    assert episode.steps == 3
    assert episode.termination_reason == "max_steps"
    assert episode.metrics["agent_steps"] == 3
    assert episode.metrics["simulator_substeps"] == 3
    assert len(client.requests) == 3
    assert episode.metrics["invalid_action_count"] == 3
    assert episode.metrics["model_api_error_count"] == 0
    assert [action.gripper for action in sim_actions] == [0.5, 0.5, 0.5]
    assert all(action.delta_position == (0.0, 0.0, 0.0) for action in sim_actions)


def test_magnifier_update_persists_and_adds_current_raw_rgb_crop(
    monkeypatch,
    tmp_path,
):
    magnifier_action = (
        _valid_action_text()[:-1]
        + ',"magnifier":{"source_rgb_key":"front",'
        '"bbox_xyxy":[0.25,0.25,0.75,0.75]}}'
    )
    client = SequencedFakeClient(
        [magnifier_action, _valid_action_text(), _valid_action_text()]
    )
    sim_actions = _capture_dummy_actions(monkeypatch)

    episode = _run_with_fake(
        monkeypatch,
        _config(tmp_path, "all"),
        client,
    )

    assert episode.steps == 3
    assert len(sim_actions) == 3
    assert [_image_count(request) for request in client.requests] == [1, 3, 5]
    second_request = client.requests[1].encoded_body().decode("utf-8")
    assert "Magnifier feedback: Magnifier activated" in second_request
    assert "Magnifier view from original source RGB key 'front'" in second_request

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text().splitlines()
    ]
    assert trajectory[0]["magnifier_update"]["bbox_xyxy"] == [
        0.25,
        0.25,
        0.75,
        0.75,
    ]
    assert trajectory[0]["magnifier_result"]["status"] == "activated"
    assert trajectory[0]["next_observation"]["magnifier"]["source_rgb_key"] == "front"
    assert trajectory[1]["next_observation"]["magnifier"]["source_rgb_key"] == "front"
    second_manifest = json.loads(
        (episode_dir / "model_calls/step_0001/request.json").read_text()
    )
    current_images = [
        image for image in second_manifest["images"] if image["step_id"] == 1
    ]
    assert [image["kind"] for image in current_images] == ["rgb", "magnifier"]


def test_bad_magnifier_key_reports_error_without_suppressing_robot_action(
    monkeypatch,
    tmp_path,
):
    channels = {f"joystick_{index}": 0 for index in range(7)}
    channels["joystick_0"] = 1
    bad_command = json.dumps(
        {
            "channels": channels,
            "magnifier": {
                "source_rgb_key": "not_a_camera",
                "bbox_xyxy": [0.1, 0.1, 0.9, 0.9],
            },
        }
    )
    config = _config(tmp_path, "all")
    config = replace(config, task=replace(config.task, max_steps=2))
    client = SequencedFakeClient([bad_command, _valid_action_text()])
    sim_actions = _capture_dummy_actions(monkeypatch)

    episode = _run_with_fake(monkeypatch, config, client)

    assert episode.steps == 2
    assert len(sim_actions) == 2
    assert sim_actions[0].delta_position[0] == pytest.approx(0.01)
    assert "unknown source_rgb_key 'not_a_camera'" in (
        client.requests[1].encoded_body().decode("utf-8")
    )
    trajectory = [
        json.loads(line)
        for line in (Path(episode.log_dir) / "trajectory.jsonl").read_text().splitlines()
    ]
    assert trajectory[0]["validation"]["is_valid"] is True
    assert trajectory[0]["magnifier_result"]["status"] == "rejected"
    assert trajectory[0]["next_observation"]["magnifier"] is None


def test_model_motion_is_clipped_then_scaled_by_the_default_unit(
    monkeypatch,
    tmp_path,
):
    response = (
        '{"channels":{"joystick_0":45,"joystick_1":-12.5,'
        '"joystick_2":0,"joystick_3":2.5,"joystick_4":0,'
        '"joystick_5":-45,"joystick_6":0.5}}'
    )
    sim_actions = _capture_dummy_actions(monkeypatch)
    config = _config(tmp_path, 0)
    config = replace(
        config,
        task=replace(config.task, max_steps=1),
    )
    episode = _run_with_fake(
        monkeypatch,
        config,
        FakeClient(response_text=response),
    )

    assert len(sim_actions) == 30
    assert episode.steps == 1
    assert episode.metrics["agent_steps"] == 1
    assert episode.metrics["simulator_steps"] == 30
    assert episode.metrics["simulator_substeps"] == 30
    assert tuple(
        sum(action.delta_position[index] for action in sim_actions)
        for index in range(3)
    ) == pytest.approx((0.3, -0.125, 0.0))
    assert tuple(
        sum(action.delta_rotation_rpy[index] for action in sim_actions)
        for index in range(3)
    ) == pytest.approx(
        (0.07853981633974475, 0.0, -0.9424777960769379)
    )
    assert all(action.gripper == 0.5 for action in sim_actions)

    trajectory = [
        json.loads(line)
        for line in (
            Path(episode.log_dir) / "trajectory.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert '"joystick_0":45' in trajectory[0]["raw_action"]
    assert trajectory[0]["parsed_action"]["joystick_0"] == 30.0
    assert trajectory[0]["parsed_action"]["joystick_1"] == -12.5
    assert trajectory[0]["parsed_action"]["joystick_5"] == -30.0
    assert trajectory[0]["sim_action"]["metadata"] == {
        "execution": "linear_motion_substeps",
        "substep_count": 30,
    }
    assert len(trajectory[0]["sim_action"]["actions"]) == 30
    assert trajectory[0]["simulator_substeps"] == 30


def test_large_commands_use_agent_step_ids_and_one_command_each(
    monkeypatch,
    tmp_path,
):
    response = (
        '{"channels":{"joystick_0":30,"joystick_1":0,'
        '"joystick_2":0,"joystick_3":0,"joystick_4":0,'
        '"joystick_5":0,"joystick_6":0.5}}'
    )
    config = _config(tmp_path, "all")
    config = replace(
        config,
        task=replace(config.task, max_steps=3, success_threshold=-1.0),
    )
    sim_actions = _capture_dummy_actions(monkeypatch)
    client = FakeClient(response_text=response)

    episode = _run_with_fake(monkeypatch, config, client)

    assert episode.termination_reason == "max_steps"
    assert episode.steps == 3
    assert episode.metrics["agent_steps"] == 3
    assert episode.metrics["simulator_substeps"] == 90
    assert len(sim_actions) == 90
    current_steps = [
        next(
            item["text"]
            for item in request.messages[-1]["content"]
            if item["type"] == "text"
        )
        for request in client.requests
    ]
    assert current_steps == [
        "Observation step 0. Camera order and available source RGB keys: front.",
        "Observation step 1. Camera order and available source RGB keys: front.",
        "Observation step 2. Camera order and available source RGB keys: front.",
    ]

    trajectory = [
        json.loads(line)
        for line in (
            Path(episode.log_dir) / "trajectory.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [transition["step_id"] for transition in trajectory] == [0, 1, 2]
    assert [
        transition["observation"]["metadata"]["simulator_step_id"]
        for transition in trajectory
    ] == [0, 30, 60]
    assert [
        transition["next_observation"]["metadata"]["simulator_step_id"]
        for transition in trajectory
    ] == [30, 60, 90]
    assert [transition["simulator_substeps"] for transition in trajectory] == [
        30,
        30,
        30,
    ]


def test_native_truncation_before_agent_budget_is_execution_budget_violation(
    monkeypatch,
    tmp_path,
):
    original_step = DummySimAdapter.step

    def truncate_early(sim, action):
        return replace(original_step(sim, action), truncated=True)

    monkeypatch.setattr(DummySimAdapter, "step", truncate_early)
    episode = _run_with_fake(
        monkeypatch,
        _config(tmp_path, 0),
        FakeClient(),
    )

    assert episode.steps == 1
    assert episode.success is False
    assert episode.termination_reason == "execution_budget_violation"
    assert episode.metrics["agent_steps"] == 1
    assert episode.metrics["simulator_substeps"] == 1


def test_model_motion_substeps_stop_when_task_succeeds(
    monkeypatch,
    tmp_path,
):
    response = (
        '{"channels":{"joystick_0":30,"joystick_1":0,'
        '"joystick_2":0,"joystick_3":0,"joystick_4":0,'
        '"joystick_5":0,"joystick_6":0.5}}'
    )
    config = _config(tmp_path, 0)
    config = replace(
        config,
        task=replace(
            config.task,
            max_steps=1,
            target_position=(0.02, 0.0, 0.2),
            success_threshold=1e-6,
        ),
    )
    sim_actions = _capture_dummy_actions(monkeypatch)

    episode = _run_with_fake(
        monkeypatch,
        config,
        FakeClient(response_text=response),
    )

    assert episode.success is True
    assert episode.termination_reason == "success"
    assert len(sim_actions) == 2
    assert episode.metrics["simulator_steps"] == 2
    assert sum(action.delta_position[0] for action in sim_actions) == pytest.approx(
        0.02
    )


def test_model_motion_substeps_stop_on_native_action_rejection(
    monkeypatch,
    tmp_path,
):
    response = (
        '{"channels":{"joystick_0":30,"joystick_1":0,'
        '"joystick_2":0,"joystick_3":0,"joystick_4":0,'
        '"joystick_5":0,"joystick_6":0.5}}'
    )
    config = _config(tmp_path, 0)
    config = replace(
        config,
        task=replace(config.task, max_steps=1),
    )
    original_step = DummySimAdapter.step
    sim_actions = []

    def rejected_step(sim, action):
        sim_actions.append(action)
        result = original_step(sim, action)
        return replace(
            result,
            info={**result.info, "action_rejected": True},
        )

    monkeypatch.setattr(DummySimAdapter, "step", rejected_step)

    _run_with_fake(
        monkeypatch,
        config,
        FakeClient(response_text=response),
    )

    assert len(sim_actions) == 1


def test_invalid_model_action_preserves_last_binary_gripper_command(
    monkeypatch,
    tmp_path,
):
    class SequenceClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.responses = [
                (
                    '{"channels":{"joystick_0":0,"joystick_1":0,'
                    '"joystick_2":0,"joystick_3":0,"joystick_4":0,'
                    '"joystick_5":0,"joystick_6":1}}'
                ),
                "{not-json",
                "{not-json",
            ]

        def complete(self, request):
            self.response_text = self.responses[len(self.requests)]
            return super().complete(request)

    config = _config(tmp_path, 0)
    config = replace(
        config,
        control_interface=replace(
            config.control_interface,
            gripper="binary",
            gripper_open_command=0.0,
            gripper_closed_command=1.0,
            gripper_hold_command=None,
        ),
    )
    sim_actions = _capture_dummy_actions(monkeypatch)
    episode = _run_with_fake(monkeypatch, config, SequenceClient())

    assert episode.metrics["invalid_action_count"] == 2
    assert [action.gripper for action in sim_actions] == [1.0, 1.0, 1.0]
    assert all(action.delta_position == (0.0, 0.0, 0.0) for action in sim_actions)
