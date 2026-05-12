from __future__ import annotations

from mllm_teleop_benchmark.action.parser import ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.control.ee_delta_pose import EEDeltaPoseController
from mllm_teleop_benchmark.core.episode import EpisodeManager
from mllm_teleop_benchmark.core.registry import build_agent, build_sim, build_task
from mllm_teleop_benchmark.core.types import BenchmarkResult, EpisodeResult
from mllm_teleop_benchmark.evaluation.evaluator import aggregate_episode_results
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.observation.camera import CameraManager
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.utils.seeding import episode_seed


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> BenchmarkResult:
        results: list[EpisodeResult] = []
        for episode_id in range(self.config.benchmark.num_episodes):
            seed = episode_seed(self.config.benchmark.seed, episode_id)
            manager = self._build_episode_manager(seed)
            results.append(manager.run_episode(episode_id=episode_id, seed=seed))
        return BenchmarkResult(episodes=results, aggregate_metrics=aggregate_episode_results(results))

    def _build_episode_manager(self, seed: int) -> EpisodeManager:
        sim = build_sim(self.config)
        task = build_task(self.config)
        camera_manager = CameraManager(self.config.camera)
        camera_manager.reset(seed)
        observation_provider = ObservationProvider(camera_manager)
        action_space = ActionSpace.from_config(self.config.action_space)
        mapping = FixedJoystickMapping.from_config(self.config.mapping)
        frame_transform = IdentityControlFrameTransform.from_config(self.config.frame_transform)
        controller = EEDeltaPoseController.from_config(self.config.controller, mapping, frame_transform)
        agent = build_agent(self.config, seed)
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
        )
