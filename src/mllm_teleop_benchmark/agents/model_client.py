from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_HIDDEN_REASONING_KEYS = {
    "analysis",
    "chain_of_thought",
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "thinking",
    "thinking_content",
    "thoughts",
}


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[dict[str, Any], ...]
    temperature: float
    max_output_tokens: int
    reasoning_effort: str | None = None
    response_format: str = "json_object"
    prompt_cache_key: str | None = None

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(self.messages),
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.prompt_cache_key:
            body["prompt_cache_key"] = self.prompt_cache_key
        return body

    def encoded_body(self) -> bytes:
        return json.dumps(self.body(), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass(frozen=True)
class ResponsesRequest:
    model: str
    instructions: str
    input: tuple[dict[str, Any], ...]
    temperature: float
    max_output_tokens: int
    reasoning_effort: str | None = None
    response_format: str = "json_object"
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    stream: bool = False

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": self.instructions,
            "input": list(self.input),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": True,
            "stream": self.stream,
        }
        if self.response_format == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if self.previous_response_id:
            body["previous_response_id"] = self.previous_response_id
        if self.prompt_cache_key:
            body["prompt_cache_key"] = self.prompt_cache_key
        return body

    def encoded_body(self) -> bytes:
        return json.dumps(self.body(), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


ModelRequestType = ModelRequest | ResponsesRequest


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None

    @property
    def cache_metrics_status(self) -> str:
        return "reported" if self.cached_input_tokens is not None else "unavailable"


@dataclass(frozen=True)
class ModelResponse:
    text: str
    request_id: str | None
    model: str | None
    usage: ModelUsage
    raw: dict[str, Any]
    latency_ms: float
    attempts: int
    request_bytes: int


class ModelClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        status_code: int | None = None,
        latency_ms: float | None = None,
        request_bytes: int = 0,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
        self.latency_ms = latency_ms
        self.request_bytes = request_bytes
        self.raw_response = raw_response


class ModelClient(ABC):
    @abstractmethod
    def complete(self, request: ModelRequestType) -> ModelResponse:
        raise NotImplementedError


HttpSender = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]


class _OpenAIHTTPClient(ModelClient):
    def __init__(
        self,
        *,
        base_url: str,
        endpoint_builder: Callable[[str], str],
        api_key: str | None,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        retry_backoff_max_seconds: float,
        retry_until_success: bool,
        sender: HttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._endpoint = endpoint_builder(base_url)
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._retry_backoff_max_seconds = retry_backoff_max_seconds
        self._retry_until_success = retry_until_success
        self._sender = sender or _send_http
        self._sleep = sleep

    def _complete(
        self,
        request: ModelRequestType,
        *,
        decode_response: Callable[[bytes], dict[str, Any]],
        response_text: Callable[[dict[str, Any]], str],
        require_response_id: bool,
    ) -> ModelResponse:
        body = request.encoded_body()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started = time.perf_counter()
        last_error = "request failed"
        attempt = 0
        while True:
            attempt += 1
            try:
                status, response_body = self._sender(
                    self._endpoint,
                    body,
                    headers,
                    self._timeout_seconds,
                )
            except (TimeoutError, socket.timeout, URLError, OSError) as exc:
                last_error = _redact_text(f"{type(exc).__name__}: {exc}", self._api_key)
                if self._should_retry(attempt):
                    self._backoff(attempt)
                    continue
                raise ModelClientError(
                    last_error,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    request_bytes=len(body),
                ) from exc

            if 200 <= status < 300:
                safe_payload: dict[str, Any] | None = None
                try:
                    payload = decode_response(response_body)
                    safe_payload = _sanitize_response(payload, self._api_key)
                    text = _redact_text(response_text(payload), self._api_key)
                    request_id = _optional_string(payload.get("id"))
                    # Responses continuation silently loses its provider context without this id.
                    if require_response_id and not request_id:
                        raise ModelClientError(
                            "Responses endpoint did not return a response id",
                            attempts=1,
                        )
                except ModelClientError as exc:
                    error_status = exc.status_code or status
                    if (
                        error_status == 429 or error_status >= 500
                    ) and self._should_retry(attempt):
                        last_error = _redact_text(str(exc), self._api_key)
                        self._backoff(attempt)
                        continue
                    raise ModelClientError(
                        str(exc),
                        attempts=attempt,
                        status_code=error_status,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        request_bytes=len(body),
                        raw_response=safe_payload,
                    ) from exc
                return ModelResponse(
                    text=text,
                    request_id=request_id,
                    model=_optional_string(payload.get("model")),
                    usage=_usage(payload.get("usage")),
                    raw=safe_payload,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=attempt,
                    request_bytes=len(body),
                )

            last_error = _redact_text(_error_message(response_body, status), self._api_key)
            if (status == 429 or status >= 500) and self._should_retry(attempt):
                self._backoff(attempt)
                continue
            raise ModelClientError(
                last_error,
                attempts=attempt,
                status_code=status,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                request_bytes=len(body),
            )

    def _backoff(self, attempt: int) -> None:
        delay = min(
            self._retry_backoff_seconds * (2 ** min(attempt - 1, 30)),
            self._retry_backoff_max_seconds,
        )
        if delay > 0:
            self._sleep(delay)

    def _should_retry(self, attempt: int) -> bool:
        return self._retry_until_success or attempt <= self._max_retries


class OpenAICompatibleClient(_OpenAIHTTPClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        retry_backoff_max_seconds: float = 30.0,
        retry_until_success: bool = False,
        sender: HttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            base_url=base_url,
            endpoint_builder=_chat_completions_endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_backoff_max_seconds=retry_backoff_max_seconds,
            retry_until_success=retry_until_success,
            sender=sender,
            sleep=sleep,
        )

    def complete(self, request: ModelRequestType) -> ModelResponse:
        return self._complete(
            request,
            decode_response=_decode_json,
            response_text=_chat_response_text,
            require_response_id=False,
        )


class OpenAIResponsesClient(_OpenAIHTTPClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        retry_backoff_max_seconds: float = 30.0,
        retry_until_success: bool = False,
        sender: HttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            base_url=base_url,
            endpoint_builder=_responses_endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_backoff_max_seconds=retry_backoff_max_seconds,
            retry_until_success=retry_until_success,
            sender=sender,
            sleep=sleep,
        )

    def complete(self, request: ModelRequestType) -> ModelResponse:
        return self._complete(
            request,
            decode_response=_decode_responses_payload,
            response_text=_responses_response_text,
            require_response_id=True,
        )


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def _responses_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/chat/completions"):
        normalized = normalized.removesuffix("/chat/completions")
    return normalized + "/responses"


def _send_http(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()


def _decode_json(payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelClientError("Model endpoint returned invalid JSON", attempts=1) from exc
    if not isinstance(decoded, dict):
        raise ModelClientError("Model endpoint returned a non-object JSON payload", attempts=1)
    return decoded


def _decode_responses_payload(payload: bytes) -> dict[str, Any]:
    stripped = payload.lstrip()
    if stripped.startswith((b"event:", b"data:", b":")):
        return _decode_responses_sse(payload)
    return _decode_json(payload)


def _decode_responses_sse(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelClientError(
            "Responses endpoint returned invalid UTF-8 SSE",
            attempts=1,
        ) from exc
    events: list[dict[str, Any]] = []
    output_items: dict[int, dict[str, Any]] = {}
    completed: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                "Responses endpoint returned invalid SSE JSON",
                attempts=1,
            ) from exc
        if not isinstance(event, dict):
            continue
        events.append(event)
        event_type = event.get("type")
        if event_type == "response.output_item.done":
            index = event.get("output_index")
            item = event.get("item")
            if isinstance(index, int) and isinstance(item, dict):
                output_items[index] = item
        elif event_type == "response.completed" and isinstance(event.get("response"), dict):
            completed = dict(event["response"])
        # Error events carry the status needed by the outer retry policy.
        elif event_type in {"error", "response.failed"}:
            error = event.get("error")
            if not isinstance(error, dict) and isinstance(event.get("response"), dict):
                error = event["response"].get("error")
            message = error.get("message") if isinstance(error, dict) else None
            status = event.get("status")
            raise ModelClientError(
                f"Responses stream error: {message or event_type}",
                attempts=1,
                status_code=_responses_sse_error_status(status, error),
            )
    # A dropped stream must become a retryable client error, not an unhandled decode failure.
    if completed is None:
        raise ModelClientError(
            "Responses stream ended without response.completed",
            attempts=1,
            status_code=503,
        )
    if not completed.get("output") and output_items:
        completed["output"] = [output_items[index] for index in sorted(output_items)]
    completed["_stream_events"] = events
    return completed


def _responses_sse_error_status(status: object, error: object) -> int | None:
    if isinstance(status, int | float):
        return int(status)
    if not isinstance(error, dict):
        return None
    error_type = str(error.get("type", "")).lower()
    error_code = str(error.get("code", "")).lower()
    if "rate_limit" in error_type or "rate_limit" in error_code:
        return 429
    if "server_error" in error_type or "server_error" in error_code:
        return 500
    return None


def _chat_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelClientError("Model response does not contain choices[0]", attempts=1)
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ModelClientError("Model response does not contain choices[0].message", attempts=1)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    raise ModelClientError("Model response message does not contain text content", attempts=1)


def _responses_response_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = payload.get("output")
    if not isinstance(output, list):
        raise ModelClientError("Responses payload does not contain output", attempts=1)
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        parts.extend(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        )
    if parts:
        return "".join(parts)
    raise ModelClientError("Responses payload does not contain output_text", attempts=1)


def _usage(value: object) -> ModelUsage:
    if not isinstance(value, dict):
        return ModelUsage()
    input_tokens = _optional_int(value.get("prompt_tokens", value.get("input_tokens")))
    output_tokens = _optional_int(value.get("completion_tokens", value.get("output_tokens")))
    details = value.get("prompt_tokens_details", value.get("input_tokens_details"))
    cached_tokens = _optional_int(details.get("cached_tokens")) if isinstance(details, dict) else None
    return ModelUsage(input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_tokens)


def _error_message(payload: bytes, status: int) -> str:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = payload.decode("utf-8", errors="replace").strip()
        return f"Model endpoint HTTP {status}: {text[:500]}" if text else f"Model endpoint HTTP {status}"
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return f"Model endpoint HTTP {status}: {error['message'][:500]}"
    return f"Model endpoint HTTP {status}"


def _sanitize_response(value: Any, secret: str | None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() == "authorization"
                else "[REDACTED: hidden reasoning]"
                if str(key).lower() in _HIDDEN_REASONING_KEYS
                else _sanitize_response(item, secret)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_response(item, secret) for item in value]
    if isinstance(value, str):
        if value.startswith("data:image/") and ";base64," in value:
            return "[image data omitted]"
        return _redact_text(value, secret)
    return value


def _redact_text(value: str, secret: str | None) -> str:
    redacted = value.replace(secret, "[REDACTED]") if secret else value
    if "data:image/" in redacted and ";base64," in redacted:
        return "[image data omitted]"
    return redacted


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
