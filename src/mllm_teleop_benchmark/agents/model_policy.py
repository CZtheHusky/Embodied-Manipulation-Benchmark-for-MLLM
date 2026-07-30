from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from typing import Any

from mllm_teleop_benchmark.agents.base import Agent, AgentContext
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelClient, ModelClientError, ModelRequest, ModelResponse
from mllm_teleop_benchmark.config.schema import AgentConfig, HistoryWindow
from mllm_teleop_benchmark.core.types import AgentArtifact, AgentDecision, AgentTrace, Observation, StepTransition
from mllm_teleop_benchmark.utils.images import encode_rgb_png, safe_artifact_name


PROMPT_VERSION = "mllm_joystick_v3"
REFLECTION_PROMPT_VERSION = "mllm_joystick_reflection_v3"
IMAGE_OMITTED = "[image omitted by history policy]"
ACTION_OMITTED = "[action omitted by history policy]"
HIDDEN_REASONING_REDACTION = "[REDACTED: hidden reasoning]"
_HIDDEN_REASONING_KEYS = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "thinking",
        "thinking_content",
        "thoughts",
    }
)


class ModelInputError(ValueError):
    pass


@dataclass(frozen=True)
class EncodedImage:
    step_id: int
    camera: str
    width: int
    height: int
    png: bytes
    sha256: str
    artifact_path: str
    data_url: str


class MLLMPolicy(Agent):
    def __init__(self, config: AgentConfig, client: ModelClient, model_name: str) -> None:
        if config.model is None:
            raise ValueError("MLLMPolicy requires agent.model configuration")
        self.config = config
        self.client = client
        self.model_name = model_name
        self.prompt_version = _prompt_version(config.action_output_mode)
        self._image_cache: dict[tuple[int, str], EncodedImage] = {}
        self._previous_message_stream = b""

    def act(self, context: AgentContext) -> AgentDecision:
        step_id = context.observation.step_id
        artifact_dir = f"model_calls/step_{step_id:04d}"
        try:
            request, images, image_count, common_prefix_bytes = self._build_request(context)
        except ModelInputError as exc:
            return AgentDecision(
                action=None,
                trace=AgentTrace(
                    actor_type="mllm",
                    status="fatal",
                    provider=self.config.model.provider,
                    api=self.config.model.api,
                    model=self.model_name,
                    reasoning_effort=self.config.model.reasoning_effort,
                    error=str(exc),
                    prompt_version=self.prompt_version,
                    artifact_dir=artifact_dir,
                ),
                artifacts=(
                    _json_artifact(
                        f"{artifact_dir}/error.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    ),
                ),
            )

        request_artifacts = self._request_artifacts(
            artifact_dir,
            request,
            images,
            common_prefix_bytes,
        )
        try:
            response = self.client.complete(request)
        except ModelClientError as exc:
            return AgentDecision(
                action=None,
                trace=AgentTrace(
                    actor_type="mllm",
                    status="error",
                    provider=self.config.model.provider,
                    api=self.config.model.api,
                    model=self.model_name,
                    reasoning_effort=self.config.model.reasoning_effort,
                    latency_ms=exc.latency_ms,
                    attempts=exc.attempts,
                    common_prefix_bytes=common_prefix_bytes,
                    request_bytes=exc.request_bytes or len(request.encoded_body()),
                    image_count=image_count,
                    error=str(exc),
                    prompt_version=self.prompt_version,
                    artifact_dir=artifact_dir,
                ),
                artifacts=(
                    *request_artifacts,
                    _json_artifact(
                        f"{artifact_dir}/error.json",
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "status_code": exc.status_code,
                            "attempts": exc.attempts,
                        },
                    ),
                ),
            )

        return AgentDecision(
            action=response.text,
            trace=self._trace(response, common_prefix_bytes, image_count, artifact_dir),
            artifacts=(
                *request_artifacts,
                _json_artifact(
                    f"{artifact_dir}/response.json",
                    sanitize_provider_response(response.raw),
                ),
            ),
        )

    def metadata(self) -> dict[str, object]:
        assert self.config.model is not None
        return {
            "type": "mllm",
            "provider": self.config.model.provider,
            "api": self.config.model.api,
            "model": self.model_name,
            "reasoning_effort": self.config.model.reasoning_effort,
            "action_output_mode": self.config.action_output_mode,
            "prompt_version": self.prompt_version,
        }

    def _build_request(
        self,
        context: AgentContext,
    ) -> tuple[ModelRequest, tuple[EncodedImage, ...], int, int]:
        assert self.config.model is not None
        cameras = self._camera_names(context.observation)
        history = context.history
        image_count = (
            1
            + sum(
                _selected(index, len(history), self.config.history.image_window)
                for index in range(len(history))
            )
        ) * len(cameras)
        if image_count > self.config.history.max_images_per_request:
            raise ModelInputError(
                "Model request contains "
                f"{image_count} images, exceeding agent.history.max_images_per_request="
                f"{self.config.history.max_images_per_request}"
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": _system_prompt(self.config.action_output_mode),
            },
            {
                "role": "user",
                "content": _static_instruction(
                    context,
                    self.prompt_version,
                    self.config.action_output_mode,
                ),
            },
        ]
        images: list[EncodedImage] = []
        for index, transition in enumerate(history):
            include_image = _selected(index, len(history), self.config.history.image_window)
            include_action = _selected(index, len(history), self.config.history.action_window)
            if not include_image and not include_action:
                continue
            observation_message, turn_images = self._observation_message(
                transition.observation,
                cameras,
                include_images=include_image,
            )
            messages.append(observation_message)
            images.extend(turn_images)
            messages.append(
                {
                    "role": "assistant",
                    "content": _history_action(
                        transition,
                        context.action_schema,
                        include_action=include_action,
                        action_output_mode=self.config.action_output_mode,
                    ),
                }
            )

        current_message, current_images = self._observation_message(
            context.observation,
            cameras,
            include_images=True,
        )
        messages.append(current_message)
        images.extend(current_images)

        unique_images = _unique_images(images)

        request = ModelRequest(
            model=self.model_name,
            messages=tuple(messages),
            temperature=self.config.model.temperature,
            max_output_tokens=self.config.model.max_output_tokens,
            reasoning_effort=self.config.model.reasoning_effort,
            response_format=self.config.model.response_format,
            prompt_cache_key=self.config.model.prompt_cache_key,
        )
        request_bytes = len(request.encoded_body())
        if request_bytes > self.config.history.max_request_bytes:
            raise ModelInputError(
                "Model request contains "
                f"{request_bytes} bytes, exceeding agent.history.max_request_bytes="
                f"{self.config.history.max_request_bytes}"
            )

        message_stream = _canonical_message_stream(request.messages)
        common_prefix_bytes = _common_prefix_length(self._previous_message_stream, message_stream)
        self._previous_message_stream = message_stream
        return request, unique_images, image_count, common_prefix_bytes

    def _camera_names(self, observation: Observation) -> tuple[str, ...]:
        configured = self.config.observation.cameras
        cameras = configured or tuple(sorted(str(name) for name in observation.images))
        if not cameras:
            raise ModelInputError("Model observation does not contain an RGB camera")
        missing = [name for name in cameras if name not in observation.images]
        if missing:
            raise ModelInputError(f"Model observation is missing configured cameras: {missing}")
        return cameras

    def _observation_message(
        self,
        observation: Observation,
        cameras: tuple[str, ...],
        *,
        include_images: bool,
    ) -> tuple[dict[str, Any], tuple[EncodedImage, ...]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Observation step {observation.step_id}. Camera order: {', '.join(cameras)}.",
            }
        ]
        images: list[EncodedImage] = []
        for camera in cameras:
            content.append({"type": "text", "text": f"Camera {camera}:"})
            if not include_images:
                content.append({"type": "text", "text": IMAGE_OMITTED})
                continue
            if camera not in observation.images:
                raise ModelInputError(
                    f"Observation step {observation.step_id} is missing configured camera {camera!r}"
                )
            encoded = self._encoded_image(observation.step_id, camera, observation.images[camera])
            images.append(encoded)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encoded.data_url},
                }
            )
        return {"role": "user", "content": content}, tuple(images)

    def _encoded_image(self, step_id: int, camera: str, value: Any) -> EncodedImage:
        key = (step_id, camera)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached

        try:
            png = encode_rgb_png(value, self.config.observation.image_max_side)
        except Exception as exc:
            raise ModelInputError(f"Failed to encode camera {camera!r} as PNG: {type(exc).__name__}") from exc
        safe_camera = safe_artifact_name(camera)
        artifact_path = f"model_calls/images/step_{step_id:04d}/{safe_camera}.png"
        encoded = EncodedImage(
            step_id=step_id,
            camera=camera,
            width=png.width,
            height=png.height,
            png=png.content,
            sha256=png.sha256,
            artifact_path=artifact_path,
            data_url="data:image/png;base64," + base64.b64encode(png.content).decode("ascii"),
        )
        self._image_cache[key] = encoded
        return encoded

    def _request_artifacts(
        self,
        artifact_dir: str,
        request: ModelRequest,
        images: tuple[EncodedImage, ...],
        common_prefix_bytes: int,
    ) -> tuple[AgentArtifact, ...]:
        manifest = [
            {
                "step_id": image.step_id,
                "camera": image.camera,
                "width": image.width,
                "height": image.height,
                "sha256": image.sha256,
                "artifact": image.artifact_path,
            }
            for image in images
        ]
        payload = {
            "prompt_version": self.prompt_version,
            "common_prefix_bytes": common_prefix_bytes,
            "request": _sanitized_request_body(request, images),
            "images": manifest,
        }
        return (
            *(AgentArtifact(image.artifact_path, image.png) for image in images),
            _json_artifact(f"{artifact_dir}/request.json", payload),
        )

    def _trace(
        self,
        response: ModelResponse,
        common_prefix_bytes: int,
        image_count: int,
        artifact_dir: str,
    ) -> AgentTrace:
        assert self.config.model is not None
        return AgentTrace(
            actor_type="mllm",
            status="ok",
            provider=self.config.model.provider,
            api=self.config.model.api,
            model=response.model or self.model_name,
            reasoning_effort=self.config.model.reasoning_effort,
            request_id=response.request_id,
            response_id=response.request_id,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            cache_hit_ratio=_cache_hit_ratio(
                response.usage.cached_input_tokens,
                response.usage.input_tokens,
            ),
            cache_metrics_status=response.usage.cache_metrics_status,
            common_prefix_bytes=common_prefix_bytes,
            request_bytes=response.request_bytes,
            image_count=image_count,
            prompt_version=self.prompt_version,
            artifact_dir=artifact_dir,
        )


def _prompt_version(action_output_mode: str) -> str:
    if action_output_mode == "action_only":
        return PROMPT_VERSION
    if action_output_mode == "action_with_reflection":
        return REFLECTION_PROMPT_VERSION
    raise ValueError(f"Unsupported agent.action_output_mode: {action_output_mode}")


def _system_prompt(action_output_mode: str = "action_only") -> str:
    prompt = (
        "You control a robot through a virtual joystick. "
        "Return exactly one JSON object and no prose outside that object. "
        "Use the current observation and the permitted history only."
    )
    if action_output_mode == "action_with_reflection":
        return (
            f"{prompt} You may include the top-level string key Reflection. "
            "When included, Reflection must concisely summarize your current "
            "hypothesis about what each joystick channel controls, based only on "
            "public observations and your own action history."
        )
    if action_output_mode != "action_only":
        raise ValueError(f"Unsupported agent.action_output_mode: {action_output_mode}")
    return prompt


def _static_instruction(
    context: AgentContext,
    prompt_version: str = PROMPT_VERSION,
    action_output_mode: str = "action_only",
) -> str:
    control = ControlInstructionBuilder().build(
        context.metadata.get("control_interface", {}),
        context.action_schema,
    )
    schema = json.dumps(context.action_schema, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    example_payload: dict[str, Any] = {
        "channels": {name: 0 for name in context.action_schema}
    }
    if action_output_mode == "action_with_reflection":
        example_payload["Reflection"] = (
            "joystick_0 appears to control ...; joystick_1 appears to control ..."
        )
    elif action_output_mode != "action_only":
        raise ValueError(f"Unsupported agent.action_output_mode: {action_output_mode}")
    example = json.dumps(
        example_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if action_output_mode == "action_with_reflection":
        format_requirement = (
            "The only permitted top-level keys are channels and the optional "
            "case-sensitive string key Reflection. "
        )
    else:
        format_requirement = (
            "The object must contain only the top-level key channels. "
        )
    return (
        f"Prompt version: {prompt_version}.\n"
        f"Task: {context.task_instruction}\n"
        f"Control: {control.text}\n"
        f"Action schema: {schema}.\n"
        f"{format_requirement}"
        f"Return valid JSON using this required output shape: {example}"
    )


def _history_action(
    transition: StepTransition,
    action_schema: dict[str, Any],
    *,
    include_action: bool,
    action_output_mode: str = "action_only",
) -> str:
    if not include_action:
        return ACTION_OMITTED
    if transition.validation.is_valid and transition.parsed_action is not None:
        payload: dict[str, Any] = {"channels": transition.parsed_action}
        if (
            action_output_mode == "action_with_reflection"
            and transition.reflection is not None
        ):
            payload["Reflection"] = transition.reflection
    else:
        status = (
            "api_error_noop"
            if transition.agent_trace is not None and transition.agent_trace.status == "error"
            else "invalid_action_noop"
        )
        payload = {
            "channels": {name: 0 for name in action_schema},
            "status": status,
        }
    if action_output_mode not in {"action_only", "action_with_reflection"}:
        raise ValueError(f"Unsupported agent.action_output_mode: {action_output_mode}")
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _selected(index: int, total: int, window: HistoryWindow) -> bool:
    if window == "all":
        return True
    count = int(window)
    return count > 0 and index >= total - count


def _canonical_message_stream(messages: tuple[dict[str, Any], ...]) -> bytes:
    return b"\n".join(
        json.dumps(message, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        for message in messages
    )


def _common_prefix_length(previous: bytes, current: bytes) -> int:
    limit = min(len(previous), len(current))
    index = 0
    while index < limit and previous[index] == current[index]:
        index += 1
    return index


def _cache_hit_ratio(cached_input_tokens: int | None, input_tokens: int | None) -> float | None:
    if cached_input_tokens is None or input_tokens is None or input_tokens <= 0:
        return None
    return cached_input_tokens / input_tokens


def _unique_images(images: list[EncodedImage]) -> tuple[EncodedImage, ...]:
    by_path: dict[str, EncodedImage] = {}
    for image in images:
        by_path.setdefault(image.artifact_path, image)
    return tuple(by_path.values())


def _sanitized_request_body(
    request: ModelRequest,
    images: tuple[EncodedImage, ...],
) -> dict[str, Any]:
    url_hashes = {image.data_url: image.sha256 for image in images}
    body = request.body()
    sanitized_messages: list[dict[str, Any]] = []
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, list):
            sanitized_messages.append(dict(message))
            continue
        sanitized_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                sanitized_content.append(dict(item) if isinstance(item, dict) else item)
                continue
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            digest = url_hashes.get(url, "unknown")
            sanitized_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;sha256={digest}"},
                }
            )
        sanitized_messages.append({**message, "content": sanitized_content})
    body["messages"] = sanitized_messages
    return body


def sanitize_provider_response(value: Any) -> Any:
    if isinstance(value, dict):
        item_type = str(value.get("type", "")).lower()
        if item_type in {"analysis", "reasoning", "thinking"}:
            return {
                "type": value.get("type"),
                "content": HIDDEN_REASONING_REDACTION,
            }
        sanitized = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            if lowered == "authorization":
                sanitized[name] = "[REDACTED]"
            elif lowered in _HIDDEN_REASONING_KEYS:
                sanitized[name] = HIDDEN_REASONING_REDACTION
            else:
                sanitized[name] = sanitize_provider_response(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_provider_response(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        return "[image data omitted]"
    return value


def _json_artifact(relative_path: str, value: object) -> AgentArtifact:
    content = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return AgentArtifact(relative_path=relative_path, content=content)
