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
    IMAGE_OMITTED,
    MLLMPolicy,
    _canonical_message_stream,
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
    action_window="all",
    max_images=128,
    max_request_bytes=67_108_864,
    action_output_mode="action_only",
):
    return AgentConfig(
        type="mllm",
        action_output_mode=action_output_mode,
        model=ModelConfig(require_api_key=False),
        observation=AgentObservationConfig(cameras=("front",), image_max_side=32),
        history=AgentHistoryConfig(
            action_window=action_window,
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


def test_json_object_requirement_is_present_in_user_input():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(), client, "fake-model")

    policy.act(_context(0))

    assert policy.prompt_version == "mllm_joystick_v2"
    request = client.requests[0]
    assert "mllm_joystick_v2" in request.messages[1]["content"]
    user_text = " ".join(
        message["content"]
        for message in request.messages
        if message["role"] == "user" and isinstance(message["content"], str)
    )
    assert "json" in user_text.lower()
    assert "Reflection" not in request.messages[0]["content"]
    assert "Reflection" not in user_text


def test_reflection_mode_changes_prompt_version_shape_and_preserves_history():
    client = RecordingClient()
    config = _agent_config(
        image_window=3,
        action_output_mode="action_with_reflection",
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

    decision = policy.act(_context(1, (transition,)))

    request = client.requests[0]
    assert request.messages[0]["role"] == "system"
    assert "top-level string key Reflection" in request.messages[0]["content"]
    static_instruction = request.messages[1]["content"]
    assert "mllm_joystick_reflection_v2" in static_instruction
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
    assert decision.trace.prompt_version == "mllm_joystick_reflection_v2"


def test_reflection_mode_includes_only_three_prior_images():
    client = RecordingClient()
    policy = MLLMPolicy(
        _agent_config(
            image_window=3,
            action_output_mode="action_with_reflection",
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
    assert _request_text(request).count(IMAGE_OMITTED) == 2
    assistant_payloads = [
        json.loads(message["content"])
        for message in request.messages
        if message["role"] == "assistant"
    ]
    assert [payload["Reflection"] for payload in assistant_payloads] == [
        "hypothesis 0",
        "hypothesis 1",
        "hypothesis 2",
        "hypothesis 3",
        "hypothesis 4",
    ]


@pytest.mark.parametrize(
    ("image_window", "expected_images", "expected_omitted"),
    [
        (0, 1, 3),
        (2, 3, 1),
        (8, 4, 0),
        ("all", 4, 0),
    ],
)
def test_image_history_window_preserves_order_and_current_frame(
    image_window,
    expected_images,
    expected_omitted,
):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=image_window), client, "fake-model")
    history = tuple(_transition(step) for step in range(3))

    policy.act(_context(3, history))

    request = client.requests[0]
    assert _image_count(request) == expected_images
    assert _request_text(request).count(IMAGE_OMITTED) == expected_omitted
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
        "Observation step 0",
        "Observation step 1",
        "Observation step 2",
        "Observation step 3",
    ]


def test_omitted_history_preserves_each_camera_label_and_replaces_only_images():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(), client, "fake-model")
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = Observation(
        step_id=0,
        images={"front": pixels, "wrist": pixels},
        state={},
    )

    included, included_images = policy._observation_message(
        observation,
        ("front", "wrist"),
        include_images=True,
    )
    omitted, omitted_images = policy._observation_message(
        observation,
        ("front", "wrist"),
        include_images=False,
    )

    included_structure = [
        IMAGE_OMITTED if item.get("type") == "image_url" else item["text"]
        for item in included["content"]
    ]
    omitted_structure = [item["text"] for item in omitted["content"]]
    assert omitted_structure == included_structure
    assert len(included_images) == 2
    assert omitted_images == ()


@pytest.mark.parametrize(
    ("trace_status", "expected_status"),
    [("error", "api_error_noop"), ("ok", "invalid_action_noop")],
)
def test_noop_history_uses_fixed_public_status_without_error_details(
    trace_status,
    expected_status,
):
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=0), client, "fake-model")
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


def test_action_window_all_keeps_every_normalized_action():
    client = RecordingClient()
    policy = MLLMPolicy(_agent_config(image_window=0, action_window="all"), client, "fake-model")
    history = tuple(_transition(step, action_value=(-1 + step)) for step in range(3))

    policy.act(_context(3, history))

    assistant_payloads = [
        json.loads(message["content"])
        for message in client.requests[0].messages
        if message["role"] == "assistant"
    ]
    assert [payload["channels"]["joystick_0"] for payload in assistant_payloads] == [-1.0, 0.0, 1.0]


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
