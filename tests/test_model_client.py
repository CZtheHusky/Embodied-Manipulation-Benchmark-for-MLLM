from dataclasses import replace
import json

import pytest

from mllm_teleop_benchmark.agents.model_client import (
    ModelClientError,
    ModelRequest,
    OpenAICompatibleClient,
    OpenAIResponsesClient,
    ResponsesRequest,
)


def _request(prompt_cache_key=None):
    return ModelRequest(
        model="test-model",
        messages=({"role": "user", "content": "test"},),
        temperature=0.0,
        max_output_tokens=16,
        prompt_cache_key=prompt_cache_key,
    )


def _response(usage=None):
    payload = {
        "id": "call-1",
        "model": "served-model",
        "choices": [{"message": {"content": "{\"channels\":{\"joystick_0\":0}}"}}],
    }
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload).encode("utf-8")


def _client(
    sender,
    sleeps=None,
    *,
    max_retries=2,
    retry_until_success=False,
    retry_backoff_max_seconds=30,
):
    return OpenAICompatibleClient(
        base_url="http://model.test/v1",
        api_key="secret",
        timeout_seconds=5,
        max_retries=max_retries,
        retry_backoff_seconds=0.25,
        retry_backoff_max_seconds=retry_backoff_max_seconds,
        retry_until_success=retry_until_success,
        sender=sender,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def _responses_request(previous_response_id=None):
    return ResponsesRequest(
        model="test-model",
        instructions="Return JSON.",
        input=(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "test"}],
            },
        ),
        temperature=0.0,
        max_output_tokens=16,
        previous_response_id=previous_response_id,
    )


def _responses_client(sender):
    return OpenAIResponsesClient(
        base_url="http://model.test/v1/chat/completions",
        api_key="secret",
        timeout_seconds=5,
        max_retries=2,
        retry_backoff_seconds=0,
        sender=sender,
        sleep=lambda _: None,
    )


def _responses_response(response_id="resp-1"):
    return json.dumps(
        {
            "id": response_id,
            "model": "served-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"channels":{"joystick_0":0}}',
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 8},
            },
        }
    ).encode("utf-8")


@pytest.mark.parametrize("first_status", [429, 500, 503])
def test_retryable_http_statuses_are_retried(first_status):
    calls = []
    sleeps = []

    def sender(url, body, headers, timeout):
        calls.append((url, body, headers, timeout))
        if len(calls) == 1:
            return first_status, b'{"error":{"message":"retry"}}'
        return 200, _response()

    response = _client(sender, sleeps).complete(_request())

    assert response.attempts == 2
    assert len(calls) == 2
    assert sleeps == [0.25]
    assert calls[0][0] == "http://model.test/v1/chat/completions"
    assert calls[0][2]["Authorization"] == "Bearer secret"


def test_timeout_is_retried():
    calls = 0

    def sender(_url, _body, _headers, _timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return 200, _response()

    assert _client(sender).complete(_request()).attempts == 2
    assert calls == 2


def test_configuration_http_error_is_not_retried():
    calls = 0

    def sender(_url, _body, _headers, _timeout):
        nonlocal calls
        calls += 1
        return 401, b'{"error":{"message":"bad credentials"}}'

    with pytest.raises(ModelClientError) as raised:
        _client(sender).complete(_request())

    assert calls == 1
    assert raised.value.attempts == 1
    assert raised.value.status_code == 401
    assert raised.value.request_bytes > 0
    assert raised.value.latency_ms is not None


def test_cached_token_usage_is_parsed_when_reported():
    def sender(_url, _body, _headers, _timeout):
        return 200, _response(
            {
                "prompt_tokens": 100,
                "completion_tokens": 7,
                "prompt_tokens_details": {"cached_tokens": 40},
            }
        )

    usage = _client(sender).complete(_request()).usage

    assert usage.input_tokens == 100
    assert usage.output_tokens == 7
    assert usage.cached_input_tokens == 40
    assert usage.cache_metrics_status == "reported"


def test_cache_metrics_are_unavailable_when_endpoint_omits_them():
    def sender(_url, _body, _headers, _timeout):
        return 200, _response({"prompt_tokens": 12, "completion_tokens": 2})

    usage = _client(sender).complete(_request()).usage

    assert usage.cached_input_tokens is None
    assert usage.cache_metrics_status == "unavailable"


def test_prompt_cache_key_is_only_sent_when_explicitly_configured():
    without_key = _request().body()
    with_key = _request("stable-episode-key").body()

    assert "prompt_cache_key" not in without_key
    assert with_key["prompt_cache_key"] == "stable-episode-key"


def test_reasoning_effort_uses_endpoint_specific_request_shape():
    chat_default = _request().body()
    chat_reasoning = replace(_request(), reasoning_effort="high").body()
    responses_default = _responses_request().body()
    responses_reasoning = replace(
        _responses_request(),
        reasoning_effort="xhigh",
    ).body()

    assert "reasoning_effort" not in chat_default
    assert chat_reasoning["reasoning_effort"] == "high"
    assert "reasoning" not in responses_default
    assert responses_reasoning["reasoning"] == {"effort": "xhigh"}


def test_response_and_error_payloads_redact_api_key_and_base64():
    payload = json.loads(_response())
    payload["debug"] = {
        "Authorization": "Bearer secret",
        "note": "echoed secret",
        "image": "data:image/png;base64,AAAA",
    }

    def success_sender(_url, _body, _headers, _timeout):
        return 200, json.dumps(payload).encode("utf-8")

    response_text = json.dumps(_client(success_sender).complete(_request()).raw)
    assert "secret" not in response_text
    assert "base64" not in response_text
    assert "[REDACTED]" in response_text

    def error_sender(_url, _body, _headers, _timeout):
        return 400, b'{"error":{"message":"echoed secret"}}'

    with pytest.raises(ModelClientError) as raised:
        _client(error_sender).complete(_request())
    assert "secret" not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


def test_retry_budget_stops_after_two_retries():
    calls = 0
    sleeps = []

    def sender(_url, _body, _headers, _timeout):
        nonlocal calls
        calls += 1
        return 503, b'{"error":{"message":"still unavailable"}}'

    with pytest.raises(ModelClientError) as raised:
        _client(sender, sleeps).complete(_request())

    assert calls == 3
    assert raised.value.attempts == 3
    assert sleeps == [0.25, 0.5]


def test_retry_until_success_ignores_retry_budget_and_caps_backoff():
    calls = 0
    sleeps = []

    def sender(_url, _body, _headers, _timeout):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return 503, b'{"error":{"message":"temporarily unavailable"}}'
        return 200, _response()

    response = _client(
        sender,
        sleeps,
        max_retries=0,
        retry_until_success=True,
        retry_backoff_max_seconds=0.5,
    ).complete(_request())

    assert response.attempts == 5
    assert calls == 5
    assert sleeps == [0.25, 0.5, 0.5, 0.5]


def test_responses_client_uses_responses_endpoint_and_chains_previous_response():
    calls = []

    def sender(url, body, headers, timeout):
        calls.append((url, json.loads(body), headers, timeout))
        return 200, _responses_response()

    response = _responses_client(sender).complete(
        _responses_request(previous_response_id="resp-0")
    )

    assert calls[0][0] == "http://model.test/v1/responses"
    assert calls[0][1]["previous_response_id"] == "resp-0"
    assert calls[0][1]["store"] is True
    assert calls[0][1]["stream"] is False
    assert calls[0][1]["text"] == {"format": {"type": "json_object"}}
    assert "messages" not in calls[0][1]
    assert response.request_id == "resp-1"
    assert response.text == '{"channels":{"joystick_0":0}}'
    assert response.usage.cached_input_tokens == 8


def test_responses_client_fails_when_response_id_is_missing():
    def sender(_url, _body, _headers, _timeout):
        return 200, _responses_response(response_id=None)

    with pytest.raises(ModelClientError, match="did not return a response id"):
        _responses_client(sender).complete(_responses_request())


def test_responses_client_collects_sse_and_restores_completed_output_items():
    text = '{"channels":{"joystick_0":0}}'
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-stream-1",
                "model": "served-model",
                "output": [],
                "usage": {"input_tokens": 30, "output_tokens": 6},
            },
        },
    ]
    payload = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    ).encode("utf-8")

    def sender(_url, body, _headers, _timeout):
        assert json.loads(body)["stream"] is True
        return 200, payload

    response = _responses_client(sender).complete(
        replace(_responses_request(), stream=True)
    )

    assert response.request_id == "resp-stream-1"
    assert response.text == text
    assert response.usage.input_tokens == 30
    assert response.raw["output"][0]["content"][0]["text"] == text
    assert len(response.raw["_stream_events"]) == 2


def test_responses_client_surfaces_sse_error_event():
    payload = (
        'event: error\ndata: {"type":"error","status":400,'
        '"error":{"message":"bad input"}}\n\n'
    ).encode("utf-8")

    def sender(_url, _body, _headers, _timeout):
        return 200, payload

    with pytest.raises(ModelClientError, match="bad input") as raised:
        _responses_client(sender).complete(
            replace(_responses_request(), stream=True)
        )

    assert raised.value.status_code == 400


def test_responses_client_retries_server_error_reported_inside_sse():
    calls = 0
    error_payload = (
        'event: error\ndata: {"type":"error",'
        '"error":{"type":"server_error","message":"temporarily unavailable"}}\n\n'
    ).encode("utf-8")

    def sender(_url, _body, _headers, _timeout):
        nonlocal calls
        calls += 1
        return (
            (200, error_payload)
            if calls == 1
            else (200, _responses_response())
        )

    response = _responses_client(sender).complete(
        replace(_responses_request(), stream=True)
    )

    assert calls == 2
    assert response.attempts == 2
