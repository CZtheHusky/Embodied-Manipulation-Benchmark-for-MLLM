from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.model_client import (
    ModelClient,
    ModelClientError,
    ModelResponse,
    ResponsesRequest,
)
from mllm_teleop_benchmark.agents.model_policy import (
    EncodedImage,
    MLLMPolicy,
    ModelInputError,
    _cache_hit_ratio,
    _history_action,
    _json_artifact,
    _static_instruction,
    _system_prompt,
    _unique_images,
    sanitize_provider_response,
)
from mllm_teleop_benchmark.config.schema import AgentConfig
from mllm_teleop_benchmark.core.types import AgentArtifact, AgentDecision, AgentTrace, Observation, StepTransition


PROMPT_VERSION = "mllm_joystick_responses_v10"


@dataclass(frozen=True)
class ResponsesBuild:
    request: ResponsesRequest
    images: tuple[EncodedImage, ...]
    image_count: int
    current_input_index: int


class ResponsesMLLMPolicy(MLLMPolicy):
    def __init__(self, config: AgentConfig, client: ModelClient, model_name: str) -> None:
        super().__init__(config, client, model_name)
        if config.model is None or config.model.api != "responses":
            raise ValueError("ResponsesMLLMPolicy requires agent.model.api=responses")
        if config.history.action_window != "all" or config.history.image_window != "all":
            raise ValueError(
                "ResponsesMLLMPolicy retains the full provider-side conversation and "
                "requires action_window=all and image_window=all"
            )
        self.prompt_version = PROMPT_VERSION
        self._previous_response_id: str | None = None
        self._last_response_step_id: int | None = None

    def act(self, context: AgentContext) -> AgentDecision:
        assert self.config.model is not None
        step_id = context.observation.step_id
        artifact_dir = f"model_calls/step_{step_id:04d}"
        try:
            build = self._build_responses_request(context)
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
                    previous_response_id=self._previous_response_id,
                    error=str(exc),
                    prompt_version=PROMPT_VERSION,
                    artifact_dir=artifact_dir,
                ),
                artifacts=(
                    _json_artifact(
                        f"{artifact_dir}/error.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    ),
                ),
            )

        request_artifacts = self._request_artifacts(artifact_dir, build)
        recovery_artifacts: tuple[AgentArtifact, ...] = ()
        recovery_count = 0
        prior_attempts = 0
        prior_latency_ms = 0.0
        prior_request_bytes = 0
        try:
            response = self.client.complete(build.request)
        except ModelClientError as exc:
            if _is_missing_previous_response(exc, build.request):
                recovery_count = 1
                prior_attempts = exc.attempts
                prior_latency_ms = exc.latency_ms or 0.0
                prior_request_bytes = exc.request_bytes
                recovery_artifacts = self._recovery_artifacts(artifact_dir, build, exc)
                self._previous_response_id = None
                self._last_response_step_id = None
                try:
                    build = self._build_responses_request(context)
                except ModelInputError as recovery_exc:
                    return AgentDecision(
                        action=None,
                        trace=AgentTrace(
                            actor_type="mllm",
                            status="fatal",
                            provider=self.config.model.provider,
                            api=self.config.model.api,
                            model=self.model_name,
                            reasoning_effort=self.config.model.reasoning_effort,
                            context_recovery_count=recovery_count,
                            error=str(recovery_exc),
                            prompt_version=PROMPT_VERSION,
                            artifact_dir=artifact_dir,
                        ),
                        artifacts=(
                            *recovery_artifacts,
                            _json_artifact(
                                f"{artifact_dir}/error.json",
                                {
                                    "type": type(recovery_exc).__name__,
                                    "message": str(recovery_exc),
                                },
                            ),
                        ),
                    )
                request_artifacts = self._request_artifacts(artifact_dir, build)
                try:
                    response = self.client.complete(build.request)
                except ModelClientError as recovery_exc:
                    exc = recovery_exc
                else:
                    exc = None
            if exc is not None:
                return AgentDecision(
                    action=None,
                    trace=AgentTrace(
                        actor_type="mllm",
                        status="error",
                        provider=self.config.model.provider,
                        api=self.config.model.api,
                        model=self.model_name,
                        reasoning_effort=self.config.model.reasoning_effort,
                        previous_response_id=build.request.previous_response_id,
                        context_recovery_count=recovery_count,
                        latency_ms=prior_latency_ms + (exc.latency_ms or 0.0),
                        attempts=prior_attempts + exc.attempts,
                        request_bytes=prior_request_bytes
                        + (exc.request_bytes or len(build.request.encoded_body())),
                        image_count=build.image_count,
                        error=str(exc),
                        prompt_version=PROMPT_VERSION,
                        artifact_dir=artifact_dir,
                    ),
                    artifacts=(
                        *recovery_artifacts,
                        *request_artifacts,
                        _json_artifact(
                            f"{artifact_dir}/error.json",
                            {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "status_code": exc.status_code,
                                "attempts": prior_attempts + exc.attempts,
                                "context_recovery_count": recovery_count,
                            },
                        ),
                    ),
                )

        if not response.request_id:
            raise RuntimeError("OpenAIResponsesClient returned no response id")
        self._previous_response_id = response.request_id
        self._last_response_step_id = step_id
        return AgentDecision(
            action=response.text,
            trace=self._trace(
                response,
                build,
                artifact_dir,
                context_recovery_count=recovery_count,
                prior_attempts=prior_attempts,
                prior_latency_ms=prior_latency_ms,
                prior_request_bytes=prior_request_bytes,
            ),
            artifacts=(
                *recovery_artifacts,
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
            "prompt_version": PROMPT_VERSION,
            "context_retention": "previous_response_id",
            "visual_tools": ["magnifier"],
        }

    def _build_responses_request(self, context: AgentContext) -> ResponsesBuild:
        assert self.config.model is not None
        cameras = self._camera_names(context.observation)
        retained_image_count = self._observation_image_count(
            context.observation,
            cameras,
        ) + sum(
            self._observation_image_count(transition.observation, cameras)
            for transition in context.history
        )
        if retained_image_count > self.config.history.max_images_per_request:
            raise ModelInputError(
                "Responses conversation retains "
                f"{retained_image_count} images, exceeding "
                "agent.history.max_images_per_request="
                f"{self.config.history.max_images_per_request}"
            )

        input_items: list[dict[str, Any]] = []
        images: list[EncodedImage] = []
        for transition in context.history:
            if self._is_uncommitted(transition):
                observation, turn_images = self._responses_observation(
                    transition.observation,
                    cameras,
                    task_instruction=context.task_instruction,
                )
                input_items.append(observation)
                images.extend(turn_images)
                input_items.append(self._execution_feedback(transition, context.action_schema))
            elif self._needs_execution_feedback(transition):
                input_items.append(self._execution_feedback(transition, context.action_schema))

        current, current_images = self._responses_observation(
            context.observation,
            cameras,
            request_action=True,
            task_instruction=context.task_instruction,
        )
        current_input_index = len(input_items)
        input_items.append(current)
        images.extend(current_images)
        unique_images = _unique_images(images)
        request = ResponsesRequest(
            model=self.model_name,
            instructions=f"{_system_prompt()}\n\n{_static_instruction(context, PROMPT_VERSION)}",
            input=tuple(input_items),
            temperature=self.config.model.temperature,
            max_output_tokens=self.config.model.max_output_tokens,
            reasoning_effort=self.config.model.reasoning_effort,
            response_format=self.config.model.response_format,
            previous_response_id=self._previous_response_id,
            prompt_cache_key=self.config.model.prompt_cache_key,
            stream=self.config.model.stream,
        )
        request_bytes = len(request.encoded_body())
        if request_bytes > self.config.history.max_request_bytes:
            raise ModelInputError(
                "Model request contains "
                f"{request_bytes} bytes, exceeding agent.history.max_request_bytes="
                f"{self.config.history.max_request_bytes}"
            )
        return ResponsesBuild(
            request=request,
            images=unique_images,
            image_count=len(images),
            current_input_index=current_input_index,
        )

    def _responses_observation(
        self,
        observation: Observation,
        cameras: tuple[str, ...],
        *,
        request_action: bool = False,
        task_instruction: str | None = None,
    ) -> tuple[dict[str, Any], tuple[EncodedImage, ...]]:
        message, images = self._observation_message(
            observation,
            cameras,
            include_images=True,
            task_instruction=task_instruction,
        )
        content = []
        for item in message["content"]:
            if item["type"] == "text":
                text = item["text"]
                if request_action and text.startswith("Observation step"):
                    text += " Respond with the required JSON joystick action."
                content.append({"type": "input_text", "text": text})
            else:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": item["image_url"]["url"],
                    }
                )
        return {"role": "user", "content": content}, images

    def _is_uncommitted(self, transition: StepTransition) -> bool:
        return (
            self._last_response_step_id is None
            or transition.step_id > self._last_response_step_id
        )

    def _needs_execution_feedback(self, transition: StepTransition) -> bool:
        return (
            self._last_response_step_id is not None
            and transition.step_id == self._last_response_step_id
            and not transition.validation.is_valid
        )

    @staticmethod
    def _execution_feedback(
        transition: StepTransition,
        action_schema: dict[str, Any],
    ) -> dict[str, Any]:
        action = _history_action(transition, action_schema, include_action=True)
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Execution feedback for step {transition.step_id}: {action}",
                }
            ],
        }

    def _request_artifacts(
        self,
        artifact_dir: str,
        build: ResponsesBuild,
    ) -> tuple[AgentArtifact, ...]:
        manifest = [
            {
                "step_id": image.step_id,
                "camera": image.camera,
                "width": image.width,
                "height": image.height,
                "sha256": image.sha256,
                "artifact": image.artifact_path,
                "kind": image.kind,
                "source_rgb_key": image.source_rgb_key,
                "bbox_xyxy": list(image.bbox_xyxy) if image.bbox_xyxy is not None else None,
            }
            for image in build.images
        ]
        payload = {
            "api": "responses",
            "prompt_version": PROMPT_VERSION,
            "common_prefix_bytes": 0,
            "current_input_index": build.current_input_index,
            "request": _sanitized_responses_request(build.request, build.images),
            "images": manifest,
        }
        return (
            *(AgentArtifact(image.artifact_path, image.png) for image in build.images),
            _json_artifact(f"{artifact_dir}/request.json", payload),
        )

    def _recovery_artifacts(
        self,
        artifact_dir: str,
        build: ResponsesBuild,
        error: ModelClientError,
    ) -> tuple[AgentArtifact, ...]:
        return (
            _json_artifact(
                f"{artifact_dir}/recovery/previous_response_not_found.json",
                {
                    "reason": "previous_response_not_found",
                    "failed_request": _sanitized_responses_request(
                        build.request,
                        build.images,
                    ),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "status_code": error.status_code,
                        "attempts": error.attempts,
                    },
                },
            ),
        )

    def _trace(
        self,
        response: ModelResponse,
        build: ResponsesBuild,
        artifact_dir: str,
        *,
        context_recovery_count: int = 0,
        prior_attempts: int = 0,
        prior_latency_ms: float = 0.0,
        prior_request_bytes: int = 0,
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
            previous_response_id=build.request.previous_response_id,
            response_id=response.request_id,
            context_recovery_count=context_recovery_count,
            latency_ms=prior_latency_ms + response.latency_ms,
            attempts=prior_attempts + response.attempts,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            cache_hit_ratio=_cache_hit_ratio(
                response.usage.cached_input_tokens,
                response.usage.input_tokens,
            ),
            cache_metrics_status=response.usage.cache_metrics_status,
            common_prefix_bytes=0,
            request_bytes=prior_request_bytes + response.request_bytes,
            image_count=build.image_count,
            prompt_version=PROMPT_VERSION,
            artifact_dir=artifact_dir,
        )


def _is_missing_previous_response(
    error: ModelClientError,
    request: ResponsesRequest,
) -> bool:
    message = str(error).lower()
    return (
        request.previous_response_id is not None
        and error.status_code in {400, 404}
        and "previous response" in message
        and "not found" in message
    )


def _sanitized_responses_request(
    request: ResponsesRequest,
    images: tuple[EncodedImage, ...],
) -> dict[str, Any]:
    url_images: dict[str, list[EncodedImage]] = {}
    for image in images:
        url_images.setdefault(image.data_url, []).append(image)
    body = request.body()
    sanitized_input: list[dict[str, Any]] = []
    for message in request.input:
        content = message.get("content")
        if not isinstance(content, list):
            sanitized_input.append(dict(message))
            continue
        sanitized_content = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "input_image":
                sanitized_content.append(dict(item) if isinstance(item, dict) else item)
                continue
            url = item.get("image_url")
            candidates = url_images.get(url, [])
            image = candidates.pop(0) if candidates else None
            digest = image.sha256 if image is not None else "unknown"
            sanitized_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;sha256={digest}",
                    "artifact": image.artifact_path if image is not None else None,
                }
            )
        sanitized_input.append({**message, "content": sanitized_content})
    body["input"] = sanitized_input
    return body
