from __future__ import annotations

from dataclasses import replace
import math

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.action.parser import ActionParseError, ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.agents.base import Agent, AgentContext
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.control.controller import Controller
from mllm_teleop_benchmark.core.execution_budget import (
    STEP_BUDGET_UNIT,
    execution_budget,
)
from mllm_teleop_benchmark.core.registry import control_interface_metadata
from mllm_teleop_benchmark.core.types import (
    CompoundSimulatorAction,
    EpisodeResult,
    Observation,
    SimulatorAction,
    StepResult,
    StepTransition,
    ValidationResult,
)
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
        run_id: str | None = None,
        run_group: str | None = None,
        source_revision: dict[str, object] | None = None,
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
        self.run_id = run_id
        self.run_group = run_group
        self.source_revision = dict(source_revision or {})
        self.execution_budget = execution_budget(config)

    def run_episode(self, episode_id: int, seed: int) -> EpisodeResult:
        logger = EpisodeLogger(
            self.config,
            episode_id,
            seed,
            run_id=self.run_id,
            run_group=self.run_group,
            source_revision=self.source_revision,
        )
        metrics = MetricsRecorder()
        history: list[StepTransition] = []

        try:
            result = self._run_episode_body(episode_id, seed, logger, metrics, history)
        finally:
            if hasattr(self.sim, "close"):
                self.sim.close()
        return result

    def _run_episode_body(
        self,
        episode_id: int,
        seed: int,
        logger: EpisodeLogger,
        metrics: MetricsRecorder,
        history: list[StepTransition],
    ) -> EpisodeResult:
        self.observation_provider.configure_sim(self.sim)
        self.task.reset(self.sim, seed)
        observation = self._agent_observation(
            self.observation_provider.get_observation(self.sim),
            agent_step_id=0,
        )
        logger.start(
            {
                "task_instruction": self.task.instruction(),
                "action_schema": self.action_space.schema(),
                "controller": self.controller.metadata(),
                "control_interface": control_interface_metadata(self.config, seed),
                "agent": self.agent.metadata(),
                "step_budget": self.execution_budget.metadata(),
            }
        )

        success = self.task.is_success(observation, self.sim.get_state())
        termination_reason = "success" if success else "max_steps"
        last_valid_action: JoystickAction | None = None
        agent_step_id = 0
        while (
            not success
            and agent_step_id < self.execution_budget.max_agent_steps
        ):
            context = AgentContext(
                observation=observation,
                task_instruction=self.task.instruction(),
                action_space=self.action_space,
                action_schema=self.action_space.schema(),
                metadata={
                    "controller": self._public_controller_metadata(),
                    "control_interface": control_interface_metadata(self.config, seed),
                    "step_budget": {
                        "unit": STEP_BUDGET_UNIT,
                        "max_agent_steps": self.execution_budget.max_agent_steps,
                    },
                },
                history=self._agent_history(history),
            )
            decision = self.agent.act(context)
            logger.record_agent_artifacts(decision.artifacts)
            metrics.update_agent_trace(decision.trace)

            if decision.trace.status in {"fatal", "error"}:
                termination_reason = "agent_error"
                break

            parsed_action, validation, reflection = self._parse_and_validate(
                decision.action
            )

            if validation.is_valid:
                action_to_execute = parsed_action
                last_valid_action = parsed_action
            else:
                action_to_execute = self._noop_joystick_action(last_valid_action)
            (
                sim_action,
                step_result,
                next_observation,
                success,
                simulator_substeps,
            ) = self._execute_action(
                action_to_execute,
            )
            next_agent_step_id = agent_step_id + 1
            next_observation = self._agent_observation(
                next_observation,
                agent_step_id=next_agent_step_id,
            )
            reached_max_steps = (
                next_agent_step_id >= self.execution_budget.max_agent_steps
            )
            truncated = reached_max_steps and not success
            transition = StepTransition(
                step_id=agent_step_id,
                observation=observation,
                raw_action=decision.action,
                parsed_action=parsed_action.as_dict() if validation.is_valid else None,
                validation=validation,
                sim_action=sim_action,
                next_observation=next_observation,
                terminated=success or (
                    step_result.terminated and not reached_max_steps
                ),
                truncated=(truncated or step_result.truncated) and not success,
                info=step_result.info,
                agent_trace=decision.trace,
                reflection=reflection,
                simulator_substeps=simulator_substeps,
            )
            logger.record_step(transition)
            metrics.update(transition)
            history.append(transition)
            observation = next_observation
            agent_step_id = next_agent_step_id

            if success:
                termination_reason = "success"
            elif reached_max_steps:
                termination_reason = "max_steps"
            elif step_result.terminated:
                termination_reason = "simulator_terminated"
            elif step_result.truncated:
                termination_reason = "execution_budget_violation"

            if transition.terminated or transition.truncated:
                break

        final_metrics = metrics.finalize(success, self.task.metrics(self.sim.get_state()))
        result = EpisodeResult(
            episode_id=episode_id,
            seed=seed,
            success=success,
            steps=agent_step_id,
            metrics=final_metrics,
            log_dir=str(logger.episode_dir),
            termination_reason=termination_reason,
        )
        logger.finish(
            result,
            transitions=tuple(history),
            terminal_observation=observation,
        )
        return result

    def _agent_history(self, history: list[StepTransition]) -> tuple[StepTransition, ...]:
        if self.config.agent.type == "mllm":
            return tuple(history)
        window = self.config.agent.history_window
        return tuple(history[-window:]) if window > 0 else ()

    def _parse_and_validate(
        self,
        raw_action: object,
    ) -> tuple[JoystickAction, ValidationResult, str | None]:
        try:
            if self.config.agent.type == "mllm":
                parsed_output = self.parser.parse_model_output(
                    raw_action,
                )
                parsed_action = parsed_output.action
                reflection = parsed_output.reflection
            else:
                parsed_action = self.parser.parse(raw_action)
                reflection = None
        except ActionParseError as exc:
            return (
                self._noop_joystick_action(),
                ValidationResult(False, (str(exc),)),
                None,
            )
        if self.config.agent.type == "mllm":
            motion_channel_count = len(
                self.config.control_interface.active_dimensions
            )
            motion_channels = self.config.control_interface.channel_names()[
                :motion_channel_count
            ]
            parsed_action = self.action_space.clip_continuous_channels(
                parsed_action,
                motion_channels,
            )
        validation = self.validator.validate(parsed_action)
        if not validation.is_valid and self.config.agent.invalid_action_recovery != "noop":
            raise ValueError(f"Unsupported invalid action recovery: {self.config.agent.invalid_action_recovery}")
        return parsed_action, validation, reflection

    def _execute_action(
        self,
        action: JoystickAction,
    ) -> tuple[SimulatorAction, StepResult, Observation, bool, int]:
        sim_actions: list[SimulatorAction] = []
        step_result: StepResult | None = None
        next_observation: Observation | None = None
        success = False
        for substep_action in self._execution_actions(action):
            translated = self.controller.translate(substep_action)
            sim_actions.append(translated)
            step_result = self.sim.step(translated)
            next_observation = self.observation_provider.get_observation(self.sim)
            success = self.task.is_success(
                next_observation,
                self.sim.get_state(),
            )
            if (
                success
                or step_result.terminated
                or step_result.truncated
                or step_result.info.get("action_rejected") is True
            ):
                break

        assert step_result is not None
        assert next_observation is not None
        if len(sim_actions) == 1:
            logged_action = sim_actions[0]
        else:
            logged_action = CompoundSimulatorAction(
                actions=tuple(sim_actions),
                metadata={
                    "execution": "linear_motion_substeps",
                    "substep_count": len(sim_actions),
                },
            )
        return (
            logged_action,
            step_result,
            next_observation,
            success,
            len(sim_actions),
        )

    def _agent_observation(
        self,
        observation: Observation,
        *,
        agent_step_id: int,
    ) -> Observation:
        metadata = dict(observation.metadata)
        metadata["simulator_step_id"] = observation.step_id
        return replace(
            observation,
            step_id=agent_step_id,
            metadata=metadata,
        )

    def _execution_actions(
        self,
        action: JoystickAction,
    ) -> tuple[JoystickAction, ...]:
        if self.config.agent.type != "mllm":
            return (action,)

        motion_channel_count = len(
            self.config.control_interface.active_dimensions
        )
        motion_channels = self.config.control_interface.channel_names()[
            :motion_channel_count
        ]
        maximum_magnitude = max(
            (
                abs(action.channels.get(channel, 0.0))
                for channel in motion_channels
            ),
            default=0.0,
        )
        substep_count = max(1, math.ceil(maximum_magnitude))
        if substep_count == 1:
            return (action,)

        substep_values = dict(action.channels)
        for channel in motion_channels:
            if channel in substep_values:
                substep_values[channel] /= substep_count
        substep = JoystickAction(substep_values)
        return (substep,) * substep_count

    def _noop_joystick_action(
        self,
        last_valid_action: JoystickAction | None = None,
    ) -> JoystickAction:
        channels = {name: 0.0 for name in self.action_space.channels}
        interface = self.config.control_interface
        if interface.gripper == "none":
            return JoystickAction(channels)

        gripper_channel = interface.channel_names()[-1]
        if interface.gripper_hold_command is not None:
            channels[gripper_channel] = interface.gripper_hold_command
        elif interface.gripper_neutral_command is not None:
            channels[gripper_channel] = interface.gripper_neutral_command
        elif (
            last_valid_action is not None
            and gripper_channel in last_valid_action.channels
        ):
            channels[gripper_channel] = last_valid_action.channels[gripper_channel]
        elif interface.gripper_open_command is not None:
            channels[gripper_channel] = interface.gripper_open_command
        return JoystickAction(channels)

    def _public_controller_metadata(self) -> dict[str, object]:
        return {
            "type": self.config.controller.type,
            "control_mode": self.config.control_interface.mode,
            "active_dimensions": list(self.config.control_interface.active_dimensions),
            "gripper": self.config.control_interface.gripper,
        }
