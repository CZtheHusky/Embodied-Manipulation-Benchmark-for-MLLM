from __future__ import annotations

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
from mllm_teleop_benchmark.logging.episode_logger import make_run_id
from mllm_teleop_benchmark.utils.seeding import episode_seed


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> BenchmarkResult:
        results: list[EpisodeResult] = []
        run_id = make_run_id()
        for episode_id in range(self.config.benchmark.num_episodes):
            seed = episode_seed(self.config.benchmark.seed, episode_id)
            manager = self._build_episode_manager(seed, run_id)
            results.append(manager.run_episode(episode_id=episode_id, seed=seed))
        return BenchmarkResult(episodes=results, aggregate_metrics=aggregate_episode_results(results))

    def _build_episode_manager(self, seed: int, run_id: str) -> EpisodeManager:
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
        )
