import json

import pytest

from mllm_teleop_benchmark.agents.model_client import (
    ModelClientError,
    ModelRequest,
    OpenAICompatibleClient,
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


def _client(sender, sleeps=None):
    return OpenAICompatibleClient(
        base_url="http://model.test/v1",
        api_key="secret",
        timeout_seconds=5,
        max_retries=2,
        retry_backoff_seconds=0.25,
        sender=sender,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


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
