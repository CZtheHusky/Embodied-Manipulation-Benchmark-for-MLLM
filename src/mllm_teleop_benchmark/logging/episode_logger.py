from __future__ import annotations

import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import AgentArtifact, EpisodeResult, Observation, StepTransition
from mllm_teleop_benchmark.logging.review_video import (
    ReviewVideoRenderer,
    _sanitize_error_text,
)
from mllm_teleop_benchmark.logging.trajectory import TrajectoryLogger
from mllm_teleop_benchmark.utils.serialization import to_jsonable


class EpisodeLogger:
    def __init__(self, config: BenchmarkConfig, episode_id: int, seed: int, run_id: str | None = None) -> None:
        self.config = config
        self.episode_id = episode_id
        self.seed = seed
        self.run_id = run_id or make_run_id()
        self.run_dir = Path(config.logging.output_dir) / self.run_id
        self.episode_dir = self.run_dir / f"episode_{episode_id:04d}"
        self.trajectory = TrajectoryLogger(self.episode_dir / "trajectory.jsonl")
        self.metadata: dict[str, Any] = {}

    def start(self, metadata: dict[str, Any]) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
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


def make_run_id(created_at: datetime | None = None) -> str:
    timestamp = created_at or datetime.now().astimezone()
    return f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
