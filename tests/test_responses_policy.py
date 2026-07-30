import json

import numpy as np

from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.model_client import (
    ModelClient,
    ModelClientError,
    ModelRequestType,
    ModelResponse,
    ModelUsage,
    ResponsesRequest,
)
from mllm_teleop_benchmark.agents.responses_policy import (
    PROMPT_VERSION,
    ResponsesMLLMPolicy,
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
    AgentDecision,
    AgentTrace,
    GymBoxAction,
    Observation,
    StepTransition,
    ValidationResult,
)


class RecordingResponsesClient(ModelClient):
    def __init__(self, outcomes=()):
        self.requests: list[ResponsesRequest] = []
        self.outcomes = list(outcomes)
        self.success_count = 0

    def complete(self, request: ModelRequestType) -> ModelResponse:
        assert isinstance(request, ResponsesRequest)
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else "valid"
        if outcome == "error":
            raise ModelClientError(
                "endpoint unavailable",
                attempts=3,
                status_code=503,
                request_bytes=len(request.encoded_body()),
            )
        if outcome == "missing_previous":
            raise ModelClientError(
                f"Previous response with id '{request.previous_response_id}' not found.",
                attempts=1,
                status_code=400,
                latency_ms=2.0,
                request_bytes=len(request.encoded_body()),
            )
        self.success_count += 1
        response_id = f"resp-{self.success_count}"
        text = (
            "not-json"
            if outcome == "invalid"
            else (
                '{"channels":{"joystick_0":0,"joystick_1":1},'
                '"Reflection":"joystick_1 appears to move the fingertip"}'
            )
        )
        raw = {
            "id": response_id,
            "model": "fake-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        }
        return ModelResponse(
            text=text,
            request_id=response_id,
            model="fake-model",
            usage=ModelUsage(input_tokens=50, output_tokens=8, cached_input_tokens=10),
            raw=raw,
            latency_ms=3.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _agent_config(max_images=128, max_request_bytes=67_108_864):
    return AgentConfig(
        type="mllm",
        model=ModelConfig(api="responses", require_api_key=False),
        observation=AgentObservationConfig(cameras=("front",), image_max_side=32),
        history=AgentHistoryConfig(
            action_window="all",
            image_window="all",
            max_images_per_request=max_images,
            max_request_bytes=max_request_bytes,
        ),
    )


def _observation(step_id):
    return Observation(
        step_id=step_id,
        images={"front": np.full((12, 16, 3), step_id * 20, dtype=np.uint8)},
        state={
            "target_position": f"PRIVATE_TARGET_{step_id}",
            "distance": f"PRIVATE_DISTANCE_{step_id}",
        },
        metadata={"seed": f"PRIVATE_SEED_{step_id}"},
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


def _transition(step_id, decision, *, valid=True):
    return StepTransition(
        step_id=step_id,
        observation=_observation(step_id),
        raw_action=decision.action,
        parsed_action=(
            {"joystick_0": 0.0, "joystick_1": 1.0}
            if valid
            else None
        ),
        validation=ValidationResult(valid, () if valid else ("private parser detail",)),
        sim_action=GymBoxAction(values=(0.0, 0.0)),
        next_observation=_observation(step_id + 1),
        terminated=False,
        truncated=False,
        agent_trace=decision.trace,
        reflection=(
            "joystick_1 appears to move the fingertip"
            if valid
            else None
        ),
    )


def _image_count(request):
    return sum(
        1
        for message in request.input
        for item in message.get("content", ())
        if isinstance(item, dict) and item.get("type") == "input_image"
    )


def test_responses_projection_excludes_privileged_state_and_uses_current_frame():
    client = RecordingResponsesClient()
    policy = ResponsesMLLMPolicy(_agent_config(), client, "fake-model")

    decision = policy.act(_context(0))

    assert PROMPT_VERSION == "mllm_joystick_responses_v6"
    assert policy.prompt_version == PROMPT_VERSION
    body = client.requests[0].encoded_body().decode("utf-8")
    assert "PRIVATE_TARGET" not in body
    assert "PRIVATE_DISTANCE" not in body
    assert "PRIVATE_SEED" not in body
    assert client.requests[0].previous_response_id is None
    assert _image_count(client.requests[0]) == 1
    assert "JSON" in json.dumps(client.requests[0].input)
    assert "optional case-sensitive string key Reflection" in client.requests[0].instructions
    assert '"Reflection"' in decision.action
    assert decision.trace.api == "responses"
    assert decision.trace.response_id == "resp-1"
    assert decision.trace.previous_response_id is None
    assert decision.trace.prompt_version == PROMPT_VERSION


def test_successful_turns_chain_by_id_without_resending_old_images():
    client = RecordingResponsesClient()
    policy = ResponsesMLLMPolicy(_agent_config(), client, "fake-model")
    first = policy.act(_context(0))
    transition = _transition(0, first)

    second = policy.act(_context(1, (transition,)))

    request = client.requests[1]
    assert request.previous_response_id == "resp-1"
    assert len(request.input) == 1
    assert "Observation step 1" in json.dumps(request.input)
    assert "Observation step 0" not in json.dumps(request.input)
    assert _image_count(request) == 1
    assert second.trace.previous_response_id == "resp-1"
    assert second.trace.response_id == "resp-2"
    assert PROMPT_VERSION in request.instructions


def test_missing_previous_response_rebuilds_full_public_history_without_stepping():
    client = RecordingResponsesClient(("valid", "missing_previous", "valid"))
    policy = ResponsesMLLMPolicy(_agent_config(), client, "fake-model")
    first = policy.act(_context(0))
    step0 = _transition(0, first)

    recovered = policy.act(_context(1, (step0,)))

    stale_request, rebuilt_request = client.requests[1:]
    assert stale_request.previous_response_id == "resp-1"
    assert rebuilt_request.previous_response_id is None
    rebuilt_body = rebuilt_request.encoded_body().decode("utf-8")
    assert "Observation step 0" in rebuilt_body
    assert "Execution feedback for step 0" in rebuilt_body
    assert "joystick_1 appears to move the fingertip" in rebuilt_body
    assert "Observation step 1" in rebuilt_body
    assert _image_count(rebuilt_request) == 2
    assert recovered.trace.status == "ok"
    assert recovered.trace.response_id == "resp-2"
    assert recovered.trace.context_recovery_count == 1
    assert recovered.trace.attempts == 2
    paths = {artifact.relative_path for artifact in recovered.artifacts}
    assert (
        "model_calls/step_0001/recovery/previous_response_not_found.json"
        in paths
    )
    recovery_artifact = next(
        artifact
        for artifact in recovered.artifacts
        if artifact.relative_path.endswith("previous_response_not_found.json")
    )
    assert "base64" not in recovery_artifact.content.decode("utf-8")


def test_api_failure_is_replayed_with_image_and_public_noop_feedback():
    client = RecordingResponsesClient(("valid", "error", "valid"))
    policy = ResponsesMLLMPolicy(_agent_config(), client, "fake-model")
    first = policy.act(_context(0))
    step0 = _transition(0, first)
    failed = policy.act(_context(1, (step0,)))
    step1 = _transition(1, failed, valid=False)

    recovered = policy.act(_context(2, (step0, step1)))

    request = client.requests[2]
    body = request.encoded_body().decode("utf-8")
    assert request.previous_response_id == "resp-1"
    assert "Observation step 1" in body
    assert "api_error_noop" in body
    assert "endpoint unavailable" not in body
    assert "Observation step 2" in body
    assert _image_count(request) == 2
    assert recovered.trace.response_id == "resp-2"


def test_invalid_response_stays_in_chain_and_gets_noop_execution_feedback():
    client = RecordingResponsesClient(("invalid", "valid"))
    policy = ResponsesMLLMPolicy(_agent_config(), client, "fake-model")
    invalid = policy.act(_context(0))
    transition = _transition(0, invalid, valid=False)

    policy.act(_context(1, (transition,)))

    request = client.requests[1]
    body = request.encoded_body().decode("utf-8")
    assert request.previous_response_id == "resp-1"
    assert "invalid_action_noop" in body
    assert "Observation step 0" not in body
    assert "Observation step 1" in body
    assert _image_count(request) == 1


def test_retained_full_context_image_limit_fails_before_encoding():
    client = RecordingResponsesClient()
    policy = ResponsesMLLMPolicy(_agent_config(max_images=2), client, "fake-model")
    history = (
        _transition(
            0,
            AgentDecision(
                action=None,
                trace=AgentTrace(actor_type="mllm", status="error"),
            ),
            valid=False,
        ),
        _transition(
            1,
            AgentDecision(
                action=None,
                trace=AgentTrace(actor_type="mllm", status="error"),
            ),
            valid=False,
        ),
    )

    decision = policy.act(_context(2, history))

    assert decision.trace.status == "fatal"
    assert "retains 3 images" in decision.trace.error
    assert client.requests == []


def test_request_artifact_records_chain_ids_without_base64():
    client = RecordingResponsesClient()
    policy = ResponsesMLLMPolicy(_agent_config(), client, "fake-model")
    first = policy.act(_context(0))
    step0 = _transition(0, first)

    second = policy.act(_context(1, (step0,)))
    request_artifact = next(
        artifact
        for artifact in second.artifacts
        if artifact.relative_path.endswith("request.json")
    )
    manifest = json.loads(request_artifact.content)

    assert manifest["api"] == "responses"
    assert manifest["current_input_index"] == 0
    assert manifest["request"]["previous_response_id"] == "resp-1"
    assert "base64" not in json.dumps(manifest)
