from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mllm_teleop_benchmark.agents.model_policy import (
    PROMPT_VERSION as CHAT_PROMPT_VERSION,
)
from mllm_teleop_benchmark.agents.responses_policy import (
    PROMPT_VERSION as RESPONSES_PROMPT_VERSION,
)
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import AgentArtifact, EpisodeResult, Observation, StepTransition
from mllm_teleop_benchmark.logging.review_video import (
    ReviewVideoRenderer,
    _sanitize_error_text,
)
from mllm_teleop_benchmark.logging.trajectory import TrajectoryLogger
from mllm_teleop_benchmark.utils.serialization import to_jsonable


class EpisodeLogger:
    def __init__(
        self,
        config: BenchmarkConfig,
        episode_id: int,
        seed: int,
        run_id: str | None = None,
        run_group: str | None = None,
        source_revision: dict[str, object] | None = None,
    ) -> None:
        self.config = config
        self.episode_id = episode_id
        self.seed = seed
        self.run_id = run_id or make_run_id()
        self.run_group = run_group
        self.source_revision = dict(source_revision or {})
        relative_run_dir = (
            Path(run_group) / self.run_id if run_group is not None else Path(self.run_id)
        )
        self.run_dir = Path(config.logging.output_dir) / relative_run_dir
        self.episode_dir = self.run_dir / f"episode_{episode_id:04d}"
        self.trajectory = TrajectoryLogger(self.episode_dir / "trajectory.jsonl")
        self.metadata: dict[str, Any] = {}

    def start(self, metadata: dict[str, Any]) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        if self.run_group is not None:
            self._write_run_manifest()
        if self.config.logging.save_config_snapshot:
            with (self.episode_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(to_jsonable(self.config.raw), handle, sort_keys=True)

    def record_step(self, transition: StepTransition) -> None:
        if self.config.logging.save_trajectory:
            self.trajectory.append(transition)

    def record_agent_artifacts(self, artifacts: tuple[AgentArtifact, ...]) -> None:
        root = self.episode_dir.resolve()
        for artifact in artifacts:
            relative = Path(artifact.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Agent artifact path must stay within the episode directory: {artifact.relative_path!r}")
            target = (self.episode_dir / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Agent artifact path escapes the episode directory: {artifact.relative_path!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != artifact.content:
                    raise RuntimeError(f"Agent artifact content changed for existing path: {artifact.relative_path}")
                continue
            target.write_bytes(artifact.content)

    def finish(
        self,
        result: EpisodeResult,
        *,
        transitions: tuple[StepTransition, ...] = (),
        terminal_observation: Observation | None = None,
    ) -> None:
        if self.config.logging.save_trajectory:
            self.trajectory.save()
        video_artifact: dict[str, Any] = {
            "status": "disabled",
            "model_view": None,
            "evaluator_debug": None,
            "manifest": None,
            "error": None,
        }
        if self.config.logging.save_video:
            try:
                if terminal_observation is None:
                    raise ValueError("Review video requires a terminal public observation")
                video_artifact = ReviewVideoRenderer(
                    self.config,
                    self.episode_dir,
                ).render(
                    transitions,
                    terminal_observation,
                    self.metadata,
                    result,
                ).summary()
            except Exception as exc:
                error = {
                    "errors": [
                        {
                            "stage": "logger",
                            "type": type(exc).__name__,
                            "message": _sanitize_error_text(str(exc)),
                        }
                    ]
                }
                review_dir = self.episode_dir / "review"
                review_dir.mkdir(parents=True, exist_ok=True)
                with (review_dir / "video_error.json").open("w", encoding="utf-8") as handle:
                    json.dump(error, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                video_artifact = {
                    "status": "failed",
                    "model_view": None,
                    "evaluator_debug": None,
                    "manifest": None,
                    "error": error,
                }
            if video_artifact["status"] == "failed":
                warnings.warn(
                    f"Review video generation failed for episode {self.episode_id}; "
                    "benchmark results and raw artifacts were preserved.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        summary = {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "success": result.success,
            "steps": result.steps,
            "termination_reason": result.termination_reason,
            "metrics": result.metrics,
            "metadata": self.metadata,
            "review_video": video_artifact,
        }
        with (self.episode_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)

    def _write_run_manifest(self) -> None:
        model = self.config.agent.model
        agent_metadata = self.metadata.get("agent", {})
        manifest = {
            "schema_version": 2,
            "run_id": self.run_id,
            "experiment_group": self.run_group,
            "source": self.source_revision,
            "benchmark": {
                "name": self.config.benchmark.name,
                "num_episodes": self.config.benchmark.num_episodes,
                "seed": self.config.benchmark.seed,
            },
            "simulator": {
                "type": self.config.simulator.type,
            },
            "task": {
                "type": self.config.task.type,
                "max_steps": self.config.task.max_steps,
            },
            "model": {
                "requested_model": (
                    agent_metadata.get("model")
                    if isinstance(agent_metadata, dict)
                    else None
                ),
                "provider": model.provider if model is not None else None,
                "api": model.api if model is not None else None,
                "reasoning_effort": model.reasoning_effort if model is not None else None,
                "stream": model.stream if model is not None else None,
                "response_format": model.response_format if model is not None else None,
                "prompt_version": (
                    agent_metadata.get("prompt_version")
                    if isinstance(agent_metadata, dict)
                    else None
                ),
            },
            "history": {
                "action_window": self.config.agent.history.action_window,
                "image_window": self.config.agent.history.image_window,
                "max_images_per_request": self.config.agent.history.max_images_per_request,
                "max_request_bytes": self.config.agent.history.max_request_bytes,
            },
            "observation": {
                "cameras": list(self.config.agent.observation.cameras),
                "image_max_side": self.config.agent.observation.image_max_side,
            },
            "camera": {
                "manager": self.config.camera.manager,
                "experiment_condition": self.config.camera.params.get(
                    "experiment_condition"
                ),
            },
            "control_interface": {
                "mode": self.config.control_interface.mode,
                "active_dimensions": list(
                    self.config.control_interface.active_dimensions
                ),
                "gripper": self.config.control_interface.gripper,
                "mapping_strategy": self.config.control_interface.mapping_strategy,
                "mapping_visible_to_agent": (
                    self.config.control_interface.mapping_visible_to_agent
                ),
                "frame_transform": self.config.control_interface.frame_transform,
                "frame_visible_to_agent": (
                    self.config.control_interface.frame_visible_to_agent
                ),
                "params": self.config.control_interface.params,
            },
            "logging": {
                "save_video": self.config.logging.save_video,
                "include_evaluator_debug": (
                    self.config.logging.video.include_evaluator_debug
                ),
            },
        }
        path = self.run_dir / "run_manifest.json"
        serialized = json.dumps(to_jsonable(manifest), indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError("Run manifest changed between episodes")
            return
        path.write_text(serialized, encoding="utf-8")


def make_run_id(created_at: datetime | None = None) -> str:
    timestamp = created_at or datetime.now().astimezone()
    return f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def make_experiment_group(config: BenchmarkConfig) -> str | None:
    if config.agent.type != "mllm" or config.agent.model is None:
        return None
    model = config.agent.model
    requested_model = model.name or os.environ.get(model.name_env) or "unresolved"
    reasoning_effort = model.reasoning_effort or "default"
    if model.api == "responses":
        history = "full"
    else:
        history = (
            f"images-{config.agent.history.image_window}"
            f"-actions-{config.agent.history.action_window}"
        )
    camera_condition = config.camera.params.get("experiment_condition")
    parts = [
        f"model-{requested_model}",
        f"reasoning-{reasoning_effort}",
        f"api-{model.api}",
        f"prompt-{_configured_prompt_version(config)}",
        f"history-{history}",
    ]
    if camera_condition is not None:
        parts.append(f"camera-{camera_condition}")
    parts.append(f"episodes-{config.benchmark.num_episodes}")
    parts.append(f"maxsteps-{config.task.max_steps}")
    if config.control_interface.mapping_strategy != "fixed":
        parts.append(f"mapping-{config.control_interface.mapping_strategy}")
    if config.control_interface.frame_transform != "identity":
        parts.append(f"frame-{config.control_interface.frame_transform}")
    return "__".join(_path_slug(part) for part in parts)


def _configured_prompt_version(config: BenchmarkConfig) -> str:
    model = config.agent.model
    if model is not None and model.api == "responses":
        return RESPONSES_PROMPT_VERSION
    return CHAT_PROMPT_VERSION


def _path_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")
    return slug or "unknown"
