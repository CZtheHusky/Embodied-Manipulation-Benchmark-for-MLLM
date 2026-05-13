from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.utils.serialization import to_jsonable


@dataclass(frozen=True)
class DebugSession:
    config: BenchmarkConfig
    actor: str
    episode_id: int
    seed: int
    run_id: str
    root_dir: Path
    env_slug: str
    control_slug: str
    actor_slug: str
    metadata: dict[str, Any]

    @classmethod
    def from_config(
        cls,
        config: BenchmarkConfig,
        *,
        actor: str,
        episode_id: int,
        seed: int,
        run_id: str | None = None,
        created_at: datetime | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> "DebugSession":
        timestamp = created_at or datetime.now().astimezone()
        resolved_run_id = run_id or f"{timestamp.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        env_name = config.simulator.env_id or config.benchmark.name
        env_slug = slugify(env_name)
        control_slug = slugify("_".join((config.control_interface.mode, *config.control_interface.active_dimensions)))
        actor_slug = slugify(actor)
        root_dir = Path(config.logging.output_dir) / "debug" / env_slug / control_slug / actor_slug / resolved_run_id
        metadata = {
            "run_id": resolved_run_id,
            "created_at": timestamp.isoformat(),
            "actor": actor,
            "actor_slug": actor_slug,
            "episode_id": int(episode_id),
            "seed": int(seed),
            "benchmark_name": config.benchmark.name,
            "simulator_type": config.simulator.type,
            "env_id": config.simulator.env_id,
            "env_slug": env_slug,
            "task_type": config.task.type,
            "control_mode": config.control_interface.mode,
            "active_dimensions": list(config.control_interface.active_dimensions),
            "control_slug": control_slug,
            "mapping_strategy": config.control_interface.mapping_strategy,
            "frame_transform": config.control_interface.frame_transform,
            "controller_type": config.controller.type,
            "controller_frame": config.controller.frame,
        }
        metadata.update(extra_metadata or {})
        return cls(
            config=config,
            actor=actor,
            episode_id=int(episode_id),
            seed=int(seed),
            run_id=resolved_run_id,
            root_dir=root_dir,
            env_slug=env_slug,
            control_slug=control_slug,
            actor_slug=actor_slug,
            metadata=metadata,
        )

    def episode_dir(self) -> Path:
        return self.root_dir / f"episode_{self.episode_id:04d}"

    def control_events_path(self) -> Path:
        return self.episode_dir() / "control_events.jsonl"

    def frame_dir(self, *, source: str = "env_render", camera_name: str = "front") -> Path:
        return self.episode_dir() / "frames" / slugify(source) / slugify(camera_name)

    def frame_path(self, *, step_id: int, prefix: str = "step", source: str = "env_render", camera_name: str = "front") -> Path:
        return self.frame_dir(source=source, camera_name=camera_name) / f"{slugify(prefix)}_{int(step_id):04d}.png"

    def event_tags(self) -> dict[str, Any]:
        return {
            "debug_run_id": self.run_id,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "actor": self.actor,
            "actor_slug": self.actor_slug,
            "env_slug": self.env_slug,
            "control_slug": self.control_slug,
        }

    def write_manifest(self) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.root_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(self.metadata), handle, indent=2, sort_keys=True)
        return path


def slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"
