from __future__ import annotations

import os
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.action.parser import ActionParseError, ActionParser
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.registry import build_action_space, build_controller, build_mapping, build_task, control_interface_metadata
from mllm_teleop_benchmark.core.types import EpisodeResult, GymBoxAction, Observation, StepResult, StepTransition, ValidationResult
from mllm_teleop_benchmark.evaluation.metrics import MetricsRecorder
from mllm_teleop_benchmark.logging.episode_logger import EpisodeLogger
from mllm_teleop_benchmark.ui.keyboard import KeyboardJoystickMapper


@dataclass(frozen=True)
class WorkerCommand:
    kind: str
    action: Any = None


@dataclass(frozen=True)
class WorkerResult:
    frame: Any
    info: dict[str, Any]
    done: bool
    state: dict[str, Any] | None = None
    step_result: StepResult | None = None
    error: str | None = None


def env_worker(config: BenchmarkConfig, action_queue: mp.Queue, res_queue: mp.Queue) -> None:
    """Isolated process for MuJoCo to avoid OpenGL context conflicts."""
    # Use EGL in its own process
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    
    import gymnasium as gym
    from mllm_teleop_benchmark.sim.gymnasium_envs.actions import action_to_native
    from mllm_teleop_benchmark.sim.gymnasium_envs.registry import (
        REGISTERED_REACHER_DELTA_EEF_ID,
        register_gymnasium_envs,
    )

    def jsonable_array(value: Any) -> Any:
        try:
            import numpy as worker_np

            if isinstance(value, worker_np.ndarray):
                return value.tolist()
            if isinstance(value, worker_np.generic):
                return value.item()
        except Exception:
            pass
        return value

    def make_state(step_id: int, native_obs: Any, reward: float | None, info: dict[str, Any]) -> dict[str, Any]:
        state = {
            "step_id": step_id,
            "env_id": config.simulator.env_id or REGISTERED_REACHER_DELTA_EEF_ID,
            "native_observation": jsonable_array(native_obs),
            "reward": reward,
        }
        state.update({key: jsonable_array(value) for key, value in info.items()})
        if "goal_distance" in state:
            state["distance_to_target"] = float(state["goal_distance"])
        return state

    def render_result(step_id: int, native_obs: Any, reward: float | None, info: dict[str, Any], done: bool, terminated: bool = False, truncated: bool = False) -> WorkerResult:
        state = make_state(step_id, native_obs, reward, info)
        observation = Observation(step_id=step_id, images={}, state=state, metadata={"simulator": "gymnasium", "env_id": state["env_id"]})
        step_result = StepResult(observation=observation, terminated=terminated, truncated=truncated, info=state)
        return WorkerResult(frame=env.render(), info=info, done=done, state=state, step_result=step_result)
    
    register_gymnasium_envs()
    env_kwargs = dict(config.simulator.env_kwargs)
    if config.simulator.frame_skip is not None:
        env_kwargs["frame_skip"] = config.simulator.frame_skip
    ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
    max_episode_steps = int(ui_config.get("max_steps", config.task.max_steps))
    
    env = gym.make(
        config.simulator.env_id or REGISTERED_REACHER_DELTA_EEF_ID,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )
    
    step_id = 0
    native_obs, info = env.reset(seed=config.benchmark.seed)
    res_queue.put(render_result(step_id, native_obs, None, dict(info), False))
    
    try:
        while True:
            cmd = action_queue.get()
            if cmd is None: # Shutdown
                break

            if isinstance(cmd, WorkerCommand):
                kind = cmd.kind
                action = cmd.action
            else:
                action, reset_req = cmd
                kind = "reset" if reset_req else "step"

            try:
                if kind == "reset":
                    step_id = 0
                    native_obs, info = env.reset(seed=config.benchmark.seed)
                    res_queue.put(render_result(step_id, native_obs, None, dict(info), False))
                elif kind == "step":
                    native_action = action_to_native(action) if isinstance(action, GymBoxAction) else action
                    native_obs, reward, terminated, truncated, info = env.step(native_action)
                    step_id += 1
                    res_queue.put(render_result(step_id, native_obs, float(reward), dict(info), terminated or truncated, bool(terminated), bool(truncated)))
                elif kind == "close":
                    break
                else:
                    raise ValueError(f"Unknown worker command kind: {kind}")
            except Exception as exc:
                res_queue.put(WorkerResult(frame=None, info={}, done=True, state=None, step_result=None, error=str(exc)))
    finally:
        env.close()

class DirectReacherPlayer:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
        self.width = int(ui_config.get("window_width", 640))
        self.image_height = int(ui_config.get("window_height", 480))
        self.panel_height = int(ui_config.get("info_panel_height", 170))
        self.height = self.image_height + self.panel_height
        self.max_steps = int(ui_config.get("max_steps", config.task.max_steps))

    def run(self) -> EpisodeResult:
        # Use simple software-backed Pygame in the main process
        os.environ["SDL_VIDEODRIVER"] = "x11"
        seed = self.config.benchmark.seed
        action_space = build_action_space(self.config)
        parser = ActionParser()
        validator = ActionValidator(action_space)
        controller = build_controller(self.config, build_mapping(self.config, seed), seed)
        task = build_task(self.config)
        key_mapper = KeyboardJoystickMapper(tuple(action_space.channels))
        logger = EpisodeLogger(self.config, 0, seed)
        logger_metadata = {
            "mode": "human_direct_worker",
            "action_schema": action_space.schema(),
            "controller": controller.metadata(),
            "control_interface": control_interface_metadata(self.config, seed),
        }
        panel_lines = ControlInstructionBuilder().build(logger_metadata["control_interface"], logger_metadata["action_schema"]).panel_lines
        metrics = MetricsRecorder()
        history: list[StepTransition] = []
        
        action_queue = mp.Queue()
        res_queue = mp.Queue()
        
        worker = mp.Process(target=env_worker, args=(self.config, action_queue, res_queue))
        worker.start()
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        font = pygame.font.Font(None, 24)
        
        steps = 0
        success = False
        last_info: dict[str, Any] = {}
        last_action = key_mapper.action_from_pressed(set())
        logger.start(logger_metadata)
        
        try:
            # Get initial frame
            initial = res_queue.get()
            if initial.error:
                raise RuntimeError(initial.error)
            frame, last_info, _ = initial.frame, initial.info, initial.done
            observation = self._observation_from_result(initial)
            self._draw(screen, font, frame, 0, None, last_action, panel_lines, success)
            pygame.display.flip()
            
            running = True
            while running and steps < self.max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
                            running = False
                        elif event.key == pygame.K_r:
                            action_queue.put(WorkerCommand("reset"))
                            result = res_queue.get()
                            if result.error:
                                raise RuntimeError(result.error)
                            frame, last_info, _ = result.frame, result.info, result.done
                            observation = self._observation_from_result(result)
                            history.clear()
                            logger = EpisodeLogger(self.config, 0, seed)
                            logger.start(logger_metadata)
                            metrics = MetricsRecorder()
                            steps = 0
                            last_action = key_mapper.action_from_pressed(set())
                            self._draw(screen, font, frame, steps, None, last_action, panel_lines, success)
                            pygame.display.flip()
                        else:
                            raw_action = key_mapper.action_from_key(pygame.key.name(event.key))
                            if raw_action is None:
                                continue
                            last_action = raw_action
                            parsed_action, validation = self._parse_and_validate(parser, validator, action_space, raw_action)
                            sim_action = controller.translate(parsed_action)
                            action_queue.put(WorkerCommand("step", sim_action))
                            result = res_queue.get()
                            if result.error:
                                raise RuntimeError(result.error)
                            frame, last_info, done = result.frame, result.info, result.done
                            steps += 1
                            next_observation = self._observation_from_result(result)
                            success = task.is_success(next_observation, next_observation.state)
                            dist = next_observation.state.get("distance_to_target", next_observation.state.get("goal_distance"))
                            truncated = steps >= self.max_steps and not success
                            step_result = result.step_result or StepResult(observation=next_observation, terminated=done, truncated=truncated, info=next_observation.state)
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
                            self._draw(screen, font, frame, steps, dist, last_action, panel_lines, success)
                            pygame.display.flip()
                            if transition.terminated or transition.truncated:
                                running = False
                time.sleep(0.01)
        finally:
            action_queue.put(WorkerCommand("close"))
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
            pygame.quit()

        final_state = observation.state if "observation" in locals() else dict(last_info)
        final_metrics = metrics.finalize(success, task.metrics(final_state))
        result = EpisodeResult(
            episode_id=0,
            seed=seed,
            success=success,
            steps=len(history),
            metrics=final_metrics,
            log_dir=str(Path(self.config.logging.output_dir) / "episode_0000"),
        )
        logger.finish(result)
        return result

    def _parse_and_validate(self, parser: ActionParser, validator: ActionValidator, action_space: Any, raw_action: dict[str, float]) -> tuple[Any, ValidationResult]:
        try:
            parsed_action = parser.parse(raw_action)
        except ActionParseError as exc:
            return self._noop_joystick_action(action_space), ValidationResult(False, (str(exc),))
        validation = validator.validate(parsed_action)
        if validation.is_valid:
            return parsed_action, validation
        return self._noop_joystick_action(action_space), validation

    def _noop_joystick_action(self, action_space: Any) -> Any:
        from mllm_teleop_benchmark.action.joystick import JoystickAction

        return JoystickAction({name: 0.0 for name in action_space.channels})

    def _observation_from_result(self, result: WorkerResult) -> Observation:
        state = dict(result.state or result.info)
        step_id = int(state.get("step_id", 0))
        return Observation(
            step_id=step_id,
            images={"front": result.frame},
            state=state,
            metadata={"simulator": "gymnasium", "source": "direct_worker"},
        )

    def _draw(self, screen, font, frame, steps, distance, action=None, panel_lines=(), success=False):
        screen.fill((18, 20, 24))
        self._draw_panel(screen, font, steps, distance, action, panel_lines, success)
        arr = np.asarray(frame)
        if arr.ndim == 3:
            if arr.shape[-1] == 4: arr = arr[..., :3]
        else:
            arr = np.zeros((self.image_height, self.width, 3), dtype=np.uint8)
        
        surface = pygame.image.frombuffer(arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGB").convert()
        if surface.get_size() != (self.width, self.image_height):
            surface = pygame.transform.scale(surface, (self.width, self.image_height))
        screen.blit(surface, (0, self.panel_height))

    def _draw_panel(self, screen, font, steps, distance, action, panel_lines, success):
        lines = [self.config.benchmark.name, f"step {steps}", f"success {success}"]
        if distance is not None:
            lines.append(f"distance {float(distance):.4f}")
        lines.extend(panel_lines)
        lines.append("keys: W/S j0  A/D j1  Q/E j2  J/L j3  K/I j4  U/O j5  R reset  Esc quit")
        if action is not None:
            action_text = " ".join(f"{name}={value:g}" for name, value in action.items())
            lines.append(f"action: {action_text}")

        y = 8
        for line in lines:
            screen.blit(font.render(line, True, (255, 255, 255)), (10, y))
            y += 18
            if y > self.panel_height - 18:
                break
