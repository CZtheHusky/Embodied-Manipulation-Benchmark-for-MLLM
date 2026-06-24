from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.action.parser import ActionParseError, ActionParser
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.registry import build_action_space, build_controller, build_mapping, build_task, control_interface_metadata
from mllm_teleop_benchmark.core.types import EpisodeResult, Observation, StepResult, StepTransition, ValidationResult
from mllm_teleop_benchmark.evaluation.metrics import MetricsRecorder
from mllm_teleop_benchmark.logging.control_debug import ControlDebugRecorder
from mllm_teleop_benchmark.logging.debug_session import DebugSession
from mllm_teleop_benchmark.logging.episode_logger import EpisodeLogger
from mllm_teleop_benchmark.logging.frame_debug import FrameDebugRecorder
from mllm_teleop_benchmark.ui.keyboard import KeyboardJoystickMapper
from mllm_teleop_benchmark.ui.runtime import PlayerRuntime, PlayerRuntimeResult


class PygameHumanPlayer:
    def __init__(self, config: BenchmarkConfig, runtime: PlayerRuntime) -> None:
        self.config = config
        self.runtime = runtime
        self.width = int(config.ui.window_width)
        self.image_height = int(config.ui.window_height)
        self.panel_height = int(config.ui.info_panel_height)
        self.height = self.image_height + self.panel_height
        self.fps = int(config.ui.fps)
        self.max_steps = int(config.ui.max_steps if config.ui.max_steps is not None else config.task.max_steps)
        self.debug_frames_enabled = bool(config.ui.debug_frames)
        self.debug_control_history_enabled = bool(config.ui.debug_control_history)

    def run(self) -> EpisodeResult:
        if self.config.ui.display:
            os.environ["DISPLAY"] = self.config.ui.display
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")

        import pygame

        seed = self.config.benchmark.seed
        action_space = build_action_space(self.config)
        parser = ActionParser()
        validator = ActionValidator(action_space)
        controller = build_controller(self.config, build_mapping(self.config, seed), seed)
        task = build_task(self.config)
        key_mapper = KeyboardJoystickMapper(tuple(action_space.channels), gripper_channel=self._gripper_channel())
        logger = EpisodeLogger(self.config, 0, seed)
        debug_session = DebugSession.from_config(
            self.config,
            actor="human_keyboard",
            episode_id=0,
            seed=seed,
            extra_metadata={"runtime": self.config.ui.runtime},
        )
        frame_debug = FrameDebugRecorder(self.config, enabled=self.debug_frames_enabled, session=debug_session)
        control_debug = ControlDebugRecorder(self.config, enabled=self.debug_control_history_enabled, session=debug_session)
        logger_metadata = {
            "mode": "human_play",
            "action_schema": action_space.schema(),
            "controller": controller.metadata(),
            "control_interface": control_interface_metadata(self.config, seed),
            "debug_session": debug_session.metadata,
            "player": {
                "backend": self.config.ui.backend,
                "runtime": self.config.ui.runtime,
                "process_model": "control_ui_main_sim_worker",
            },
        }
        panel_lines = ControlInstructionBuilder().build(logger_metadata["control_interface"], logger_metadata["action_schema"]).panel_lines
        metrics = MetricsRecorder()
        history: list[StepTransition] = []

        self.runtime.start()
        pygame.init()
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()

        steps = 0
        success = False
        last_action = key_mapper.action_from_pressed(set())
        logger.start(logger_metadata)
        control_debug.reset()
        if self.debug_frames_enabled and not self.debug_control_history_enabled:
            debug_session.write_manifest()

        try:
            initial = self.runtime.reset()
            frame, observation = self._frame_observation(initial)
            self._configure_canvas_from_frame(frame)
            screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.config.benchmark.name)
            frame_debug.record_frame(frame, step_id=observation.step_id, prefix="initial")
            self._draw(screen, font, frame, steps, None, last_action, panel_lines, success)
            pygame.display.flip()

            running = True
            while running and steps < self.max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        key_name = pygame.key.name(event.key)
                        if event.key in (pygame.K_ESCAPE, pygame.K_p):
                            control_debug.record(
                                control_debug.build_event_record(
                                    event="quit",
                                    source="human_keyboard",
                                    control_id=key_name,
                                    step_before=steps,
                                    state_before=observation.state,
                                )
                            )
                            running = False
                        elif event.key == pygame.K_r:
                            result = self.runtime.reset()
                            frame, next_observation = self._frame_observation(result)
                            before_state = observation.state
                            observation = next_observation
                            frame_debug.record_frame(frame, step_id=observation.step_id, prefix="reset")
                            control_debug.record(
                                control_debug.build_event_record(
                                    event="reset",
                                    source="human_keyboard",
                                    control_id=key_name,
                                    step_before=steps,
                                    state_before=before_state,
                                    state_after=observation.state,
                                )
                            )
                            history.clear()
                            logger = EpisodeLogger(self.config, 0, seed)
                            logger.start(logger_metadata)
                            metrics = MetricsRecorder()
                            steps = 0
                            success = False
                            last_action = key_mapper.action_from_pressed(set())
                            self._draw(screen, font, frame, steps, None, last_action, panel_lines, success)
                            pygame.display.flip()
                        else:
                            raw_action = key_mapper.action_from_key(key_name)
                            if raw_action is None:
                                continue
                            before_observation = observation
                            last_action = raw_action
                            parsed_action, validation = self._parse_and_validate(parser, validator, action_space, raw_action)
                            sim_action = controller.translate(parsed_action)
                            result = self.runtime.step(sim_action)
                            frame, next_observation = self._frame_observation(result)
                            steps += 1
                            frame_debug.record_frame(frame, step_id=next_observation.step_id, prefix="step")
                            control_record = control_debug.build_step_record(
                                source="human_keyboard",
                                control_id=key_name,
                                step_before=steps - 1,
                                raw_action=raw_action,
                                validation=validation,
                                sim_action=sim_action,
                                state_before=before_observation.state,
                                state_after=next_observation.state,
                            )
                            control_debug.record(control_record)
                            success = task.is_success(next_observation, next_observation.state)
                            dist = self._display_distance(next_observation.state)
                            truncated = steps >= self.max_steps and not success
                            step_result = result.step_result or StepResult(
                                observation=next_observation,
                                terminated=result.terminated,
                                truncated=result.truncated,
                                info=next_observation.state,
                            )
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
                                info={**step_result.info, "control_debug": control_record},
                            )
                            logger.record_step(transition)
                            metrics.update(transition)
                            history.append(transition)
                            observation = next_observation
                            self._draw(screen, font, frame, steps, dist, last_action, panel_lines, success)
                            pygame.display.flip()
                            if transition.terminated or transition.truncated:
                                running = False
                clock.tick(self.fps)
                time.sleep(0.001)
        finally:
            self.runtime.close()
            pygame.quit()

        final_state = observation.state if "observation" in locals() else {}
        final_metrics = metrics.finalize(success, task.metrics(final_state))
        result = EpisodeResult(
            episode_id=0,
            seed=seed,
            success=success,
            steps=len(history),
            metrics=final_metrics,
            log_dir=str(logger.episode_dir),
        )
        logger.finish(result)
        return result

    def _frame_observation(self, result: PlayerRuntimeResult) -> tuple[Any, Observation]:
        if result.error:
            raise RuntimeError(result.error)
        if result.observation is None:
            raise RuntimeError("Player runtime returned no observation")
        return result.frame, result.observation

    def _parse_and_validate(self, parser: ActionParser, validator: ActionValidator, action_space: Any, raw_action: dict[str, float]) -> tuple[JoystickAction, ValidationResult]:
        try:
            parsed_action = parser.parse(raw_action)
        except ActionParseError as exc:
            return self._noop_joystick_action(action_space), ValidationResult(False, (str(exc),))
        validation = validator.validate(parsed_action)
        if validation.is_valid:
            return parsed_action, validation
        return self._noop_joystick_action(action_space), validation

    def _noop_joystick_action(self, action_space: Any) -> JoystickAction:
        return JoystickAction({name: 0.0 for name in action_space.channels})

    def _draw(self, screen: Any, font: Any, frame: Any, steps: int, distance: Any, action: dict[str, float] | None = None, panel_lines: tuple[str, ...] = (), success: bool = False) -> None:
        import pygame

        screen.fill((18, 20, 24))
        self._draw_panel(screen, font, steps, distance, action, panel_lines, success)
        arr = self._rgb_array(frame)
        surface = pygame.image.frombuffer(arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGB").convert()
        if surface.get_size() != (self.width, self.image_height):
            surface = pygame.transform.scale(surface, (self.width, self.image_height))
        screen.blit(surface, (0, self.panel_height))

    def _draw_panel(self, screen: Any, font: Any, steps: int, distance: Any, action: dict[str, float] | None, panel_lines: tuple[str, ...], success: bool) -> None:
        lines = [self.config.benchmark.name, f"step {steps}", f"success {success}"]
        if distance is not None:
            lines.append(f"distance {float(distance):.4f}")
        lines.extend(panel_lines)
        lines.append(f"keys: {self._key_help(action)}  R reset  Esc quit")
        if action is not None:
            action_text = " ".join(f"{name}={value:g}" for name, value in action.items())
            lines.append(f"action: {action_text}")

        y = 8
        for line in lines:
            screen.blit(font.render(line, True, (255, 255, 255)), (10, y))
            y += 18
            if y > self.panel_height - 18:
                break

    def _key_help(self, action: dict[str, float] | None) -> str:
        if action is None:
            channel_count = len(self.config.action_space.channels)
        else:
            channel_count = len(action)
        labels = ["W/S j0", "A/D j1", "Q/E j2", "J/L j3", "K/I j4", "U/O j5"]
        help_items = labels[: min(channel_count, len(labels))]
        if self._gripper_channel() is not None:
            help_items.append("G/Space gripper")
        return "  ".join(help_items)

    def _gripper_channel(self) -> str | None:
        if self.config.control_interface.gripper == "none":
            return None
        channels = self.config.control_interface.channel_names()
        return channels[-1] if channels else None

    def _display_distance(self, state: dict[str, Any]) -> Any:
        for key in ("distance_to_target", "goal_distance", "object_goal_distance"):
            if key in state:
                return state[key]
        return None

    def _configure_canvas_from_frame(self, frame: Any) -> None:
        size = self._frame_size(frame)
        if size is None:
            self.height = self.image_height + self.panel_height
            return
        frame_width, frame_height = size
        self.width = frame_width
        self.image_height = frame_height
        self.height = self.image_height + self.panel_height

    def _frame_size(self, frame: Any) -> tuple[int, int] | None:
        try:
            array = self._rgb_array(frame)
            height, width = array.shape[:2]
            if int(width) <= 0 or int(height) <= 0:
                return None
            return int(width), int(height)
        except Exception:
            return None

    def _rgb_array(self, frame: Any) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3:
            if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.moveaxis(arr, 0, -1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[-1] == 3:
                if arr.dtype != np.uint8:
                    if float(np.nanmax(arr)) <= 1.0:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(arr)
        return np.zeros((self.image_height, self.width, 3), dtype=np.uint8)


__all__ = ["PygameHumanPlayer"]
