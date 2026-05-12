from __future__ import annotations

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.action.parser import ActionParseError, ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.agents.base import Agent, AgentContext
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.types import EpisodeResult, SimAction, StepTransition, ValidationResult
from mllm_teleop_benchmark.evaluation.metrics import MetricsRecorder
from mllm_teleop_benchmark.logging.episode_logger import EpisodeLogger
from mllm_teleop_benchmark.observation.provider import ObservationProvider
from mllm_teleop_benchmark.sim.adapter import SimAdapter
from mllm_teleop_benchmark.tasks.base import Task


class EpisodeManager:
    def __init__(
        self,
        config: BenchmarkConfig,
        sim: SimAdapter,
        task: Task,
        observation_provider: ObservationProvider,
        action_space: ActionSpace,
        parser: ActionParser,
        validator: ActionValidator,
        controller: Controller,
        agent: Agent,
    ) -> None:
        self.config = config
        self.sim = sim
        self.task = task
        self.observation_provider = observation_provider
        self.action_space = action_space
        self.parser = parser
        self.validator = validator
        self.controller = controller
        self.agent = agent

    def run_episode(self, episode_id: int, seed: int) -> EpisodeResult:
        logger = EpisodeLogger(self.config, episode_id, seed)
        metrics = MetricsRecorder()
        history: list[StepTransition] = []

        self.task.reset(self.sim, seed)
        observation = self.observation_provider.get_observation(self.sim)
        logger.start(
            {
                "task_instruction": self.task.instruction(),
                "action_schema": self.action_space.schema(),
                "controller": self.controller.metadata(),
                "agent": self.agent.metadata(),
            }
        )

        success = self.task.is_success(observation, self.sim.get_state())
        while not success and len(history) < self.config.task.max_steps:
            context = AgentContext(
                observation=observation,
                task_instruction=self.task.instruction(),
                action_space=self.action_space,
                action_schema=self.action_space.schema(),
                metadata={"controller": self.controller.metadata()},
                history=tuple(history[-self.config.agent.history_window :]),
            )
            raw_action = self.agent.act(context)
            parsed_action, validation = self._parse_and_validate(raw_action)
            sim_action = self.controller.translate(parsed_action) if validation.is_valid else self._noop_action()
            step_result = self.sim.step(sim_action)
            next_observation = self.observation_provider.get_observation(self.sim)
            success = self.task.is_success(next_observation, self.sim.get_state())
            truncated = len(history) + 1 >= self.config.task.max_steps and not success
            transition = StepTransition(
                step_id=observation.step_id,
                observation=observation,
                raw_action=raw_action,
                parsed_action=parsed_action.as_dict() if validation.is_valid else None,
                validation=validation,
                sim_action=sim_action,
                next_observation=next_observation,
                terminated=success or step_result.terminated,
                truncated=truncated or step_result.truncated,
                info=step_result.info,
            )
            logger.record_step(transition)
            metrics.update(transition)
            history.append(transition)
            observation = next_observation
            if transition.terminated or transition.truncated:
                break

        final_metrics = metrics.finalize(success, self.task.metrics(self.sim.get_state()))
        result = EpisodeResult(
            episode_id=episode_id,
            seed=seed,
            success=success,
            steps=len(history),
            metrics=final_metrics,
            log_dir=str(logger.episode_dir),
        )
        logger.finish(result)
        return result

    def _parse_and_validate(self, raw_action: object) -> tuple[JoystickAction, ValidationResult]:
        try:
            parsed_action = self.parser.parse(raw_action)
        except ActionParseError as exc:
            return self._noop_joystick_action(), ValidationResult(False, (str(exc),))
        validation = self.validator.validate(parsed_action)
        if not validation.is_valid and self.config.agent.invalid_action_recovery != "noop":
            raise ValueError(f"Unsupported invalid action recovery: {self.config.agent.invalid_action_recovery}")
        return parsed_action, validation

    def _noop_joystick_action(self) -> JoystickAction:
        return JoystickAction({name: 0.0 for name in self.action_space.channels})

    def _noop_action(self) -> SimAction:
        return SimAction(delta_position=(0.0, 0.0, 0.0), delta_rotation_rpy=(0.0, 0.0, 0.0), gripper=0.0)
