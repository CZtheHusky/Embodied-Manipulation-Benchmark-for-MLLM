from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

from mllm_teleop_benchmark.action.parser import ActionParser
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.episode import EpisodeManager
from mllm_teleop_benchmark.core.registry import (
    build_agent,
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    build_task,
)
from mllm_teleop_benchmark.core.types import BenchmarkResult, EpisodeResult
from mllm_teleop_benchmark.evaluation.evaluator import aggregate_episode_results
from mllm_teleop_benchmark.logging.episode_logger import (
    make_experiment_group,
    make_run_id,
)
from mllm_teleop_benchmark.utils.seeding import episode_seed
from mllm_teleop_benchmark.utils.serialization import to_jsonable
from mllm_teleop_benchmark.utils.source_revision import capture_source_revision


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self, episode_ids: Iterable[int] | None = None) -> BenchmarkResult:
        results: list[EpisodeResult] = []
        run_id = make_run_id()
        run_group = make_experiment_group(self.config)
        source_revision = capture_source_revision(self.config.source_path)
        selected_episode_ids = self._selected_episode_ids(episode_ids)
        for episode_id in selected_episode_ids:
            seed = episode_seed(self.config.benchmark.seed, episode_id)
            manager = self._build_episode_manager(
                seed,
                run_id,
                run_group,
                source_revision,
            )
            results.append(manager.run_episode(episode_id=episode_id, seed=seed))
        aggregate_metrics = aggregate_episode_results(results)
        if run_group is not None:
            self._write_run_summary(run_id, run_group, results, aggregate_metrics)
        return BenchmarkResult(episodes=results, aggregate_metrics=aggregate_metrics)

    def _selected_episode_ids(
        self,
        episode_ids: Iterable[int] | None,
    ) -> tuple[int, ...]:
        if episode_ids is None:
            return tuple(range(self.config.benchmark.num_episodes))
        selected = tuple(episode_ids)
        if not selected:
            raise ValueError("At least one episode id must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError("Episode ids must be unique")
        invalid = [
            episode_id
            for episode_id in selected
            if not isinstance(episode_id, int)
            or isinstance(episode_id, bool)
            or episode_id < 0
            or episode_id >= self.config.benchmark.num_episodes
        ]
        if invalid:
            raise ValueError(
                "Episode ids must be integers in "
                f"[0, {self.config.benchmark.num_episodes}): {invalid}"
            )
        return selected

    def _build_episode_manager(
        self,
        seed: int,
        run_id: str,
        run_group: str | None,
        source_revision: dict[str, object],
    ) -> EpisodeManager:
        agent = build_agent(self.config, seed)
        sim = build_sim(self.config)
        task = build_task(self.config)
        camera_manager = build_camera_manager(self.config.camera, seed)
        observation_provider = build_observation_provider(self.config, camera_manager)
        action_space = build_action_space(self.config)
        mapping = build_mapping(self.config, seed)
        controller = build_controller(self.config, mapping, seed)
        parser = ActionParser()
        validator = ActionValidator(action_space)
        return EpisodeManager(
            config=self.config,
            sim=sim,
            task=task,
            observation_provider=observation_provider,
            action_space=action_space,
            parser=parser,
            validator=validator,
            controller=controller,
            agent=agent,
            run_id=run_id,
            run_group=run_group,
            source_revision=source_revision,
        )

    def _write_run_summary(
        self,
        run_id: str,
        run_group: str,
        results: list[EpisodeResult],
        aggregate_metrics: dict[str, object],
    ) -> None:
        run_dir = Path(self.config.logging.output_dir) / run_group / run_id
        observed_models: set[str] = set()
        for result in results:
            trajectory_path = Path(result.log_dir) / "trajectory.jsonl"
            if not trajectory_path.is_file():
                continue
            for line in trajectory_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                value = json.loads(line)
                trace = value.get("agent_trace") if isinstance(value, dict) else None
                model = trace.get("model") if isinstance(trace, dict) else None
                if isinstance(model, str) and model:
                    observed_models.add(model)
        summary = {
            "run_id": run_id,
            "experiment_group": run_group,
            "observed_response_models": sorted(observed_models),
            "aggregate_metrics": aggregate_metrics,
            "episodes": [
                {
                    "episode_id": result.episode_id,
                    "seed": result.seed,
                    "success": result.success,
                    "steps": result.steps,
                    "termination_reason": result.termination_reason,
                    "log_dir": str(Path(result.log_dir).relative_to(run_dir)),
                }
                for result in results
            ],
        }
        with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(summary), handle, indent=2, sort_keys=True)
            handle.write("\n")
