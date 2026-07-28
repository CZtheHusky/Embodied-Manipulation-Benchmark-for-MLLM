from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import socket
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: tuple[dict[str, Any], ...]
    temperature: float
    max_output_tokens: int
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
        if self.prompt_cache_key:
            body["prompt_cache_key"] = self.prompt_cache_key
        return body

    def encoded_body(self) -> bytes:
        return json.dumps(self.body(), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


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
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
        self.latency_ms = latency_ms
        self.request_bytes = request_bytes


class ModelClient(ABC):
    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError


HttpSender = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]


class OpenAICompatibleClient(ModelClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        sender: HttpSender | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None:
            raise ValueError("OpenAI-compatible base URL must be an absolute http(s) URL")
        self._endpoint = _chat_completions_endpoint(base_url)
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sender = sender or _send_http
        self._sleep = sleep

    def complete(self, request: ModelRequest) -> ModelResponse:
        body = request.encoded_body()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started = time.perf_counter()
        last_error = "request failed"
        last_status: int | None = None
        for attempt in range(1, self._max_retries + 2):
            try:
                status, response_body = self._sender(
                    self._endpoint,
                    body,
                    headers,
                    self._timeout_seconds,
                )
            except (TimeoutError, socket.timeout, URLError, OSError) as exc:
                last_error = _redact_text(f"{type(exc).__name__}: {exc}", self._api_key)
                if attempt <= self._max_retries:
                    self._backoff(attempt)
                    continue
                raise ModelClientError(
                    last_error,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    request_bytes=len(body),
                ) from exc

            last_status = status
            if 200 <= status < 300:
                try:
                    payload = _decode_json(response_body)
                    response_text = _redact_text(_response_text(payload), self._api_key)
                except ModelClientError as exc:
                    raise ModelClientError(
                        str(exc),
                        attempts=attempt,
                        status_code=status,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        request_bytes=len(body),
                    ) from exc
                safe_payload = _sanitize_response(payload, self._api_key)
                return ModelResponse(
                    text=response_text,
                    request_id=_optional_string(payload.get("id")),
                    model=_optional_string(payload.get("model")),
                    usage=_usage(payload.get("usage")),
                    raw=safe_payload,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=attempt,
                    request_bytes=len(body),
                )

            last_error = _redact_text(_error_message(response_body, status), self._api_key)
            if (status == 429 or status >= 500) and attempt <= self._max_retries:
                self._backoff(attempt)
                continue
            raise ModelClientError(
                last_error,
                attempts=attempt,
                status_code=status,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                request_bytes=len(body),
            )

        raise ModelClientError(
            last_error,
            attempts=self._max_retries + 1,
            status_code=last_status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            request_bytes=len(body),
        )

    def _backoff(self, attempt: int) -> None:
        delay = self._retry_backoff_seconds * (2 ** (attempt - 1))
        if delay > 0:
            self._sleep(delay)


def _chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


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


def _response_text(payload: dict[str, Any]) -> str:
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
            str(key): "[REDACTED]" if str(key).lower() == "authorization" else _sanitize_response(item, secret)
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
