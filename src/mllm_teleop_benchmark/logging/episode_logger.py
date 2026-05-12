from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import EpisodeResult, StepTransition
from mllm_teleop_benchmark.logging.trajectory import TrajectoryLogger
from mllm_teleop_benchmark.utils.serialization import to_jsonable


class EpisodeLogger:
    def __init__(self, config: BenchmarkConfig, episode_id: int, seed: int) -> None:
        self.config = config
        self.episode_id = episode_id
        self.seed = seed
        self.episode_dir = Path(config.logging.output_dir) / f"episode_{episode_id:04d}"
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

    def finish(self, result: EpisodeResult) -> None:
        if self.config.logging.save_trajectory:
            self.trajectory.save()
        summary = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "success": result.success,
            "steps": result.steps,
            "metrics": result.metrics,
            "metadata": self.metadata,
        }
        with (self.episode_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
