from dataclasses import replace
import json

import numpy as np
import pytest

import mllm_teleop_benchmark.agents.model_policy as policy_module
import mllm_teleop_benchmark.utils.images as image_utils
from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.model_client import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from mllm_teleop_benchmark.agents.model_policy import (
    HIDDEN_REASONING_REDACTION,
    MLLMPolicy,
    _canonical_message_stream,
    sanitize_provider_response,
)
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import (
    AgentConfig,
    AgentHistoryConfig,
    AgentObservationConfig,
    ModelConfig,
)
from mllm_teleop_benchmark.core.registry import build_action_space, control_interface_metadata
from mllm_teleop_benchmark.core.types import (
    AgentTrace,
    GymBoxAction,
    MagnifierUpdate,
    MagnifierView,
    Observation,
    StepTransition,
    ValidationResult,
)


class RecordingClient(ModelClient):
    def __init__(self):
        self.requests = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        text = '{"channels":{"joystick_0":0,"joystick_1":1}}'
        raw = {
            "id": f"request-{len(self.requests)}",
            "model": "fake-model",
            "choices": [{"message": {"content": text}}],
        }
        return ModelResponse(
            text=text,
            request_id=raw["id"],
            model=raw["model"],
            usage=ModelUsage(input_tokens=100, output_tokens=8, cached_input_tokens=25),
            raw=raw,
            latency_ms=4.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _agent_config(
    image_window=2,
    action_window=None,
    max_images=128,
    max_request_bytes=67_108_864,
):
    return AgentConfig(
        type="mllm",
        model=ModelConfig(require_api_key=False),
        observation=AgentObservationConfig(cameras=("front",), image_max_side=32),
        history=AgentHistoryConfig(
            action_window=image_window if action_window is None else action_window,
            image_window=image_window,
            max_images_per_request=max_images,
            max_request_bytes=max_request_bytes,
        ),
    )


def _observation(step_id):
    return Observation(
        step_id=step_id,
        images={"front": np.full((12, 16, 3), (step_id * 30) % 256, dtype=np.uint8)},
        state={
            "target_position": f"PRIVATE_TARGET_{step_id}",
            "distance": f"PRIVATE_DISTANCE_{step_id}",
            "reward": f"PRIVATE_REWARD_{step_id}",
            "seed": f"PRIVATE_SEED_{step_id}",
        },
        metadata={"camera_azimuth": f"PRIVATE_CAMERA_RANDOMIZATION_{step_id}"},
    )


def _transition(step_id, action_value=0):
    observation = _observation(step_id)
    return StepTransition(
        step_id=step_id,
        observation=observation,
        raw_action={"private_raw_action": f"PRIVATE_RAW_{step_id}"},
        parsed_action={"joystick_0": float(action_value), "joystick_1": 0.0},
        validation=ValidationResult(True),
        sim_action=GymBoxAction(
            values=(0.0, 0.0),
            metadata={"simulator_secret": f"PRIVATE_SIM_ACTION_{step_id}"},
        ),
        next_observation=_observation(step_id + 1),
        terminated=False,
        truncated=False,
        info={"reward": f"PRIVATE_INFO_{step_id}", "success": True},
    )


def _context(step_id, history=()):
    benchmark_config = load_config("configs/reacher_delta_eef.yaml")
    action_space = build_action_space(benchmark_config)
    return AgentContext(
        observation=_observation(step_id),
        task_instruction="Move the visible fingertip to the visible target.",
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={"control_interface": control_interface_metadata(benchmark_config, seed=7)},
        history=tuple(history),
    )


def _image_count(request):
    count = 0
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
    return count


def _request_text(request):
    return request.encoded_body().decode("utf-8")


def test_projection_excludes_all_privileged_simulator_fields():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window="all"), client, "fake-model")

    policy.act(_context(1, (_transition(0),)))

    request_text = _request_text(client.requests[0])
    for forbidden in (
        "PRIVATE_TARGET",
        "PRIVATE_DISTANCE",
        "PRIVATE_REWARD",
        "PRIVATE_SEED",
        "PRIVATE_CAMERA_RANDOMIZATION",
        "PRIVATE_RAW",
        "PRIVATE_SIM_ACTION",
        "PRIVATE_INFO",
    ):
        assert forbidden not in request_text
    assert "Move the visible fingertip" in request_text
    assert "joystick_0" in request_text


@pytest.mark.parametrize(
    ("agent_step", "expects_reminder"),
    [(0, False), (9, False), (10, True), (11, False), (20, True)],
)
def test_task_instruction_is_repeated_every_ten_agent_steps(
    agent_step,
    expects_reminder,
):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(), client, "fake-model")

    policy.act(_context(agent_step))

    current_observation = client.requests[0].messages[-1]["content"]
    current_text = " ".join(
        item["text"]
        for item in current_observation
        if item.get("type") == "text"
    )
    reminder = f"Task reminder at agent step {agent_step}:"
    assert (reminder in current_text) is expects_reminder
    if expects_reminder:
        assert "Move the visible fingertip to the visible target." in current_text


def test_task_reminder_is_preserved_when_the_agent_step_becomes_history():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window="all"), client, "fake-model")

    policy.act(_context(11, (_transition(10),)))

    historical_observation = client.requests[0].messages[2]["content"]
    historical_text = " ".join(
        item["text"]
        for item in historical_observation
        if item.get("type") == "text"
    )
    assert "Task reminder at agent step 10:" in historical_text


def test_json_object_requirement_is_present_in_user_input():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(), client, "fake-model")

    policy.act(_context(0))

    assert policy.prompt_version == "mllm_joystick_chat_completions_v11"
    request = client.requests[0]
    assert "mllm_joystick_chat_completions_v11" in request.messages[1]["content"]
    user_text = " ".join(
        message["content"]
        for message in request.messages
        if message["role"] == "user" and isinstance(message["content"], str)
    )
    assert "json" in user_text.lower()
    assert "top-level string key Reflection" in request.messages[0]["content"]
    assert "any information you believe will help with future decisions" in (
        request.messages[0]["content"]
    )
    assert "what each joystick channel controls" not in request.messages[0]["content"]
    assert "optional case-sensitive string key Reflection" in user_text
    assert "Magnifier tool" in user_text
    assert "source_rgb_key" in user_text
    assert "bbox_xyxy" in user_text


def test_response_artifact_sanitization_removes_hidden_reasoning():
    raw = {
        "choices": [
            {
                "message": {
                    "content": "{\"channels\":{}}",
                    "reasoning_content": "PRIVATE_CHAIN_OF_THOUGHT",
                }
            }
        ],
        "output": [
            {
                "type": "reasoning",
                "id": "reasoning-item",
                "summary": "PRIVATE_REASONING_SUMMARY",
                "encrypted_content": "PRIVATE_ENCRYPTED_REASONING",
            }
        ],
        "usage": {
            "completion_tokens_details": {
                "reasoning_tokens": 42,
            }
        },
    }

    sanitized = sanitize_provider_response(raw)
    serialized = json.dumps(sanitized)

    assert "PRIVATE_CHAIN_OF_THOUGHT" not in serialized
    assert "PRIVATE_REASONING_SUMMARY" not in serialized
    assert "PRIVATE_ENCRYPTED_REASONING" not in serialized
    assert (
        sanitized["choices"][0]["message"]["reasoning_content"]
        == HIDDEN_REASONING_REDACTION
    )
    assert sanitized["output"][0] == {
        "type": "reasoning",
        "content": HIDDEN_REASONING_REDACTION,
    }
    assert (
        sanitized["usage"]["completion_tokens_details"]["reasoning_tokens"]
        == 42
    )


def test_chat_completions_contract_and_history_preserve_reflection():
    client = RecordingClient()
    config = _agent_config(
        image_window=3,
    )
    policy = MLLMPolicy(config, client, "fake-model")
    transition = replace(
        _transition(0, action_value=1),
        raw_action=(
            '{"channels":{"joystick_0":1,"joystick_1":0},'
            '"Reflection":"joystick_0 appears to move right"}'
        ),
        reflection="joystick_0 appears to move right",
    )

    context = replace(
        _context(1, (transition,)),
        task_instruction="pick coke can",
    )
    decision = policy.act(context)

    request = client.requests[0]
    assert request.messages[0]["role"] == "system"
    assert "top-level string key Reflection" in request.messages[0]["content"]
    static_instruction = request.messages[1]["content"]
    assert "mllm_joystick_chat_completions_v11" in static_instruction
    assert "\nTask: pick coke can\nControl:" in static_instruction
    assert "optional case-sensitive string key Reflection" in static_instruction
    assistant_payload = json.loads(
        next(
            message["content"]
            for message in request.messages
            if message["role"] == "assistant"
        )
    )
    assert assistant_payload == {
        "Reflection": "joystick_0 appears to move right",
        "channels": {"joystick_0": 1.0, "joystick_1": 0.0},
    }
    assert decision.trace.prompt_version == "mllm_joystick_chat_completions_v11"


def test_chat_completions_includes_only_three_prior_images():
    client = RecordingClient()
    policy = MLLMPolicy(
        _agent_config(
            image_window=3,
        ),
        client,
        "fake-model",
    )
    history = tuple(
        replace(_transition(step), reflection=f"hypothesis {step}")
        for step in range(5)
    )

    policy.act(_context(5, history))

    request = client.requests[0]
    assert _image_count(request) == 4
    assistant_payloads = [
        json.loads(message["content"])
        for message in request.messages
        if message["role"] == "assistant"
    ]
    assert [payload["Reflection"] for payload in assistant_payloads] == [
        "hypothesis 2",
        "hypothesis 3",
        "hypothesis 4",
    ]


@pytest.mark.parametrize(
    ("image_window", "expected_steps"),
    [
        (0, [3]),
        (2, [1, 2, 3]),
        (8, [0, 1, 2, 3]),
        ("all", [0, 1, 2, 3]),
    ],
)
def test_history_window_preserves_paired_turns_and_current_frame(
    image_window,
    expected_steps,
):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=image_window), client, "fake-model")
    history = tuple(_transition(step) for step in range(3))

    policy.act(_context(3, history))

    request = client.requests[0]
    assert _image_count(request) == len(expected_steps)
    observation_labels = [
        item["text"]
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict)
        and item.get("type") == "text"
        and item.get("text", "").startswith("Observation step")
    ]
    assert [label.split(".", 1)[0] for label in observation_labels] == [
        f"Observation step {step}" for step in expected_steps
    ]
    assistant_messages = [
        message for message in request.messages if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == len(expected_steps) - 1


def test_current_observation_includes_magnifier_image_feedback_and_manifest():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(), client, "fake-model")
    observation = replace(
        _observation(1),
        magnifier=MagnifierView(
            source_rgb_key="front",
            bbox_xyxy=(0.2, 0.3, 0.7, 0.9),
            image=np.full((12, 8, 3), 77, dtype=np.uint8),
        ),
        tool_feedback=("Magnifier activated.",),
    )

    decision = policy.act(replace(_context(1), observation=observation))

    request = client.requests[0]
    assert _image_count(request) == 2
    request_text = _request_text(request)
    assert "available source RGB keys: front" in request_text
    assert "Magnifier feedback: Magnifier activated." in request_text
    assert "normalized bbox_xyxy=[0.2,0.3,0.7,0.9]" in request_text
    manifest = json.loads(
        next(
            artifact.content
            for artifact in decision.artifacts
            if artifact.relative_path.endswith("request.json")
        )
    )
    current = [image for image in manifest["images"] if image["step_id"] == 1]
    assert [image["kind"] for image in current] == ["rgb", "magnifier"]
    assert current[1]["source_rgb_key"] == "front"
    assert current[1]["bbox_xyxy"] == [0.2, 0.3, 0.7, 0.9]
    assert any(
        artifact.relative_path.endswith("magnifier/front.png")
        for artifact in decision.artifacts
    )


def test_history_replays_normalized_magnifier_command():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=1), client, "fake-model")
    transition = replace(
        _transition(0),
        magnifier_update=MagnifierUpdate(
            requested=True,
            source_rgb_key="front",
            bbox_xyxy=(0.1, 0.2, 0.8, 0.9),
        ),
    )

    policy.act(_context(1, (transition,)))

    assistant = next(
        json.loads(message["content"])
        for message in client.requests[0].messages
        if message["role"] == "assistant"
    )
    assert assistant["magnifier"] == {
        "source_rgb_key": "front",
        "bbox_xyxy": [0.1, 0.2, 0.8, 0.9],
    }


def test_magnifier_artifact_namespace_cannot_collide_with_camera_key():
    policy = MLLMPolicy(_agent_config(), RecordingClient(), "fake-model")
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = Observation(
        step_id=0,
        images={"magnifier_front": pixels},
        state={},
        magnifier=MagnifierView(
            source_rgb_key="magnifier_front",
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
            image=pixels,
        ),
    )

    _message, images = policy._observation_message(
        observation,
        ("magnifier_front",),
    )

    assert [image.artifact_path for image in images] == [
        "model_calls/images/step_0000/magnifier_front.png",
        "model_calls/images/step_0000/magnifier/magnifier_front.png",
    ]


@pytest.mark.parametrize(
    ("trace_status", "expected_status"),
    [("error", "api_error_noop"), ("ok", "invalid_action_noop")],
)
def test_noop_history_uses_fixed_public_status_without_error_details(
    trace_status,
    expected_status,
):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=1), client, "fake-model")
    transition = replace(
        _transition(0),
        parsed_action=None,
        validation=ValidationResult(False, ("PRIVATE_ENDPOINT_ERROR",)),
        agent_trace=AgentTrace(
            actor_type="mllm",
            status=trace_status,
            error="PRIVATE_ENDPOINT_ERROR",
        ),
    )

    policy.act(_context(1, (transition,)))

    request = client.requests[0]
    assistant_payload = next(
        json.loads(message["content"])
        for message in request.messages
        if message["role"] == "assistant"
    )
    assert assistant_payload == {
        "channels": {"joystick_0": 0, "joystick_1": 0},
        "status": expected_status,
    }
    assert "PRIVATE_ENDPOINT_ERROR" not in _request_text(request)


def test_history_window_drops_actions_without_selected_images():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=2), client, "fake-model")
    history = tuple(_transition(step, action_value=(-1 + step)) for step in range(3))

    policy.act(_context(3, history))

    assistant_payloads = [
        json.loads(message["content"])
        for message in client.requests[0].messages
        if message["role"] == "assistant"
    ]
    assert [payload["channels"]["joystick_0"] for payload in assistant_payloads] == [0.0, 1.0]


def test_all_history_keeps_previous_canonical_stream_as_exact_prefix():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window="all"), client, "fake-model")

    first = policy.act(_context(0))
    second = policy.act(_context(1, (_transition(0),)))

    first_stream = _canonical_message_stream(client.requests[0].messages)
    second_stream = _canonical_message_stream(client.requests[1].messages)
    assert second_stream.startswith(first_stream)
    assert first.trace.common_prefix_bytes == 0
    assert second.trace.common_prefix_bytes == len(first_stream)


def test_history_images_are_encoded_once_and_reused_byte_for_byte(monkeypatch):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window="all"), client, "fake-model")
    original_imwrite = image_utils.iio.imwrite
    encode_calls = []

    def recording_imwrite(*args, **kwargs):
        encode_calls.append(args)
        return original_imwrite(*args, **kwargs)

    monkeypatch.setattr(image_utils.iio, "imwrite", recording_imwrite)
    first = policy.act(_context(0))
    second = policy.act(_context(1, (_transition(0),)))

    assert len(encode_calls) == 2
    first_png = next(artifact.content for artifact in first.artifacts if artifact.relative_path.endswith("front.png"))
    repeated_png = next(
        artifact.content
        for artifact in second.artifacts
        if artifact.relative_path == "model_calls/images/step_0000/front.png"
    )
    assert first_png == repeated_png
    first_manifest = json.loads(
        next(artifact.content for artifact in first.artifacts if artifact.relative_path.endswith("request.json"))
    )
    second_manifest = json.loads(
        next(artifact.content for artifact in second.artifacts if artifact.relative_path.endswith("request.json"))
    )
    assert first_manifest["images"][0]["sha256"] == second_manifest["images"][0]["sha256"]
    assert "base64" not in json.dumps(second_manifest)


@pytest.mark.parametrize(
    ("config", "error_fragment"),
    [
        (_agent_config(image_window="all", max_images=1), "max_images_per_request=1"),
        (_agent_config(image_window=0, max_request_bytes=1), "max_request_bytes=1"),
    ],
)
def test_request_safety_limits_fail_loud_without_calling_endpoint(config, error_fragment):
    client = RecordingClient()
    policy = MLLMPolicy(config, client, "fake-model")
    history = (_transition(0),) if config.history.image_window == "all" else ()

    decision = policy.act(_context(len(history), history))

    assert decision.trace.status == "fatal"
    assert error_fragment in decision.trace.error
    assert client.requests == []


def test_cached_token_metrics_are_recorded_without_claiming_client_prefix_as_hit():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(), client, "fake-model")

    decision = policy.act(_context(0))

    assert decision.trace.cached_input_tokens == 25
    assert decision.trace.cache_hit_ratio == 0.25
    assert decision.trace.cache_metrics_status == "reported"
    assert decision.trace.common_prefix_bytes == 0


def test_image_count_limit_is_checked_before_png_encoding(monkeypatch):
    client = RecordingClient()
    policy = MLLMPolicy(
        _agent_config(image_window="all", max_images=1),
        client,
        "fake-model",
    )

    def fail_if_encoded(*_args, **_kwargs):
        raise AssertionError("image limit must be checked before PNG encoding")

    monkeypatch.setattr(image_utils.iio, "imwrite", fail_if_encoded)
    decision = policy.act(_context(1, (_transition(0),)))

    assert decision.trace.status == "fatal"
    assert "max_images_per_request=1" in decision.trace.error
    assert client.requests == []


def test_magnifier_counts_toward_image_limit_before_png_encoding(monkeypatch):
    client = RecordingClient()
    policy = MLLMPolicy(
        _agent_config(image_window=0, max_images=1),
        client,
        "fake-model",
    )
    observation = replace(
        _observation(0),
        magnifier=MagnifierView(
            source_rgb_key="front",
            bbox_xyxy=(0.0, 0.0, 0.5, 0.5),
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
    )

    def fail_if_encoded(*_args, **_kwargs):
        raise AssertionError("magnifier image limit must be checked before encoding")

    monkeypatch.setattr(image_utils.iio, "imwrite", fail_if_encoded)
    decision = policy.act(replace(_context(0), observation=observation))

    assert decision.trace.status == "fatal"
    assert "contains 2 images" in decision.trace.error
    assert "max_images_per_request=1" in decision.trace.error
    assert client.requests == []


def test_default_image_limit_rejects_129_images_before_encoding(monkeypatch):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window="all"), client, "fake-model")
    history = tuple(_transition(step) for step in range(128))

    def fail_if_encoded(*_args, **_kwargs):
        raise AssertionError("129-image request must fail before encoding")

    monkeypatch.setattr(image_utils.iio, "imwrite", fail_if_encoded)
    decision = policy.act(_context(128, history))

    assert decision.trace.status == "fatal"
    assert "129 images" in decision.trace.error
    assert "max_images_per_request=128" in decision.trace.error
    assert client.requests == []


def test_default_byte_limit_rejects_64_mib_plus_one(monkeypatch):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=0), client, "fake-model")

    class OversizedPayload:
        def __len__(self):
            return 67_108_865

    monkeypatch.setattr(
        policy_module.ModelRequest,
        "encoded_body",
        lambda _request: OversizedPayload(),
    )
    decision = policy.act(_context(0))

    assert decision.trace.status == "fatal"
    assert "67108865 bytes" in decision.trace.error
    assert "max_request_bytes=67108864" in decision.trace.error
    assert client.requests == []
