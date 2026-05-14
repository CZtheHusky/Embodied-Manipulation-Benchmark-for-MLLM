from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.action.parser import ActionParser
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.registry import build_action_space, build_camera_manager, build_controller, build_mapping, build_observation_provider, build_sim, build_task, control_interface_metadata
from mllm_teleop_benchmark.core.types import EpisodeResult, StepTransition
from mllm_teleop_benchmark.evaluation.metrics import MetricsRecorder
from mllm_teleop_benchmark.logging.episode_logger import EpisodeLogger
from mllm_teleop_benchmark.ui.keyboard import KeyboardJoystickMapper


class PygameBenchmarkApp:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
        self.width = int(ui_config.get("window_width", 960))
        self.image_height = int(ui_config.get("window_height", 640))
        self.panel_height = int(ui_config.get("info_panel_height", 170))
        self.height = self.image_height + self.panel_height
        self.fps = int(ui_config.get("fps", 20))
        self.max_steps = int(ui_config.get("max_steps", config.task.max_steps))
        self.simple_blit = bool(ui_config.get("simple_blit", True))
        self.debug_frames = bool(ui_config.get("debug_frames", False))
        self.mujoco_gl = str(ui_config.get("mujoco_gl", "egl"))
        self._last_valid_frame: Any = None
        self._panel_lines: tuple[str, ...] = ()

    def run(self) -> EpisodeResult:
        os.environ["MUJOCO_GL"] = self.mujoco_gl
        if self.mujoco_gl == "egl":
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        print(f"[play_benchmark] MUJOCO_GL: {os.environ['MUJOCO_GL']}")
        print(f"[play_benchmark] PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM', '')}")
        seed = self.config.benchmark.seed
        sim = build_sim(self.config)
        task = build_task(self.config)
        camera_manager = build_camera_manager(self.config.camera, seed)
        observation_provider = build_observation_provider(self.config, camera_manager)
        action_space = build_action_space(self.config)
        parser = ActionParser()
        validator = ActionValidator(action_space)
        controller = build_controller(self.config, build_mapping(self.config, seed), seed)
        key_mapper = KeyboardJoystickMapper(tuple(action_space.channels))
        logger = EpisodeLogger(self.config, 0, seed)
        metrics = MetricsRecorder()
        history: list[StepTransition] = []

        observation_provider.configure_sim(sim)
        task.reset(sim, seed)
        observation = observation_provider.get_observation(sim)
        control_metadata = control_interface_metadata(self.config, seed)
        logger.start({"mode": "human_pygame", "action_schema": action_space.schema(), "controller": controller.metadata(), "control_interface": control_metadata})
        self._panel_lines = ControlInstructionBuilder().build(control_metadata, action_space.schema()).panel_lines

        import pygame

        pygame.display.init()
        pygame.font.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.config.benchmark.name)
        print(f"[play_benchmark] SDL video driver: {pygame.display.get_driver()}")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)

        running = True
        success = False
        last_action = key_mapper.action_from_pressed(set())
        self._draw(screen, font, observation, last_action, success)
        self._present(pygame)
        self._save_debug_screen(screen, "initial")
        while running and len(history) < self.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        key_mapper.toggle_gripper()
                    elif event.key == pygame.K_r:
                        task.reset(sim, seed)
                        observation = observation_provider.get_observation(sim)
                        history.clear()
                        last_action = key_mapper.action_from_pressed(set())
                        self._draw(screen, font, observation, last_action, success)
                        self._present(pygame)
                        self._save_debug_screen(screen, "reset")
                    else:
                        raw_action = key_mapper.action_from_key(pygame.key.name(event.key))
                        if raw_action is None:
                            continue
                        last_action = raw_action
                        observation, success, running = self._step_once(
                            sim=sim,
                            task=task,
                            observation_provider=observation_provider,
                            parser=parser,
                            validator=validator,
                            controller=controller,
                            logger=logger,
                            metrics=metrics,
                            history=history,
                            observation=observation,
                            raw_action=raw_action,
                            success=success,
                            running=running,
                        )
                        self._draw(screen, font, observation, last_action, success)
                        self._present(pygame)
                        self._save_debug_screen(screen, f"step_{len(history):04d}")
            clock.tick(self.fps)

        result = EpisodeResult(
            episode_id=0,
            seed=seed,
            success=success,
            steps=len(history),
            metrics=metrics.finalize(success, task.metrics(sim.get_state())),
            log_dir=str(Path(self.config.logging.output_dir) / "episode_0000"),
        )
        logger.finish(result)
        if hasattr(sim, "close"):
            sim.close()
        pygame.quit()
        return result

    def _draw(self, screen: Any, font: Any, observation: Any, action: dict[str, float], success: bool) -> None:
        import pygame

        screen.fill((24, 26, 30))
        frame = next(iter(observation.images.values()), None)
        if self._is_valid_frame(frame):
            self._last_valid_frame = frame
        elif self._last_valid_frame is not None:
            frame = self._last_valid_frame
        if frame is not None:
            surface = self._frame_to_surface(frame, pygame)
            if surface is not None:
                if self.simple_blit:
                    self._draw_panel(screen, font, observation, action, success)
                    scaled = pygame.transform.scale(surface, (self.width, self.image_height))
                    screen.blit(scaled, (0, self.panel_height))
                    self._draw_overlay(screen, font, observation, action, success)
                    return
                surface = pygame.transform.smoothscale(surface, (600, 600))
                screen.blit(surface, (16, 16))
        x = 640
        lines = [
            self.config.benchmark.name,
            f"step: {observation.step_id}",
            f"mode: keydown steps ({self.max_steps} max)",
            f"distance: {float(observation.state.get('distance_to_target', 0.0)):.4f}",
            f"success: {success}",
            "",
            "Keys:",
            "W/S: joystick_0",
            "A/D: joystick_1",
            "Q/E: joystick_2",
            "Space: gripper",
            "R: reset  Esc: quit",
            "",
            "Action:",
        ]
        for channel, value in action.items():
            lines.append(f"{channel}: {value:g}")
        y = 24
        for line in lines:
            color = (230, 232, 235) if line else (150, 150, 150)
            screen.blit(font.render(line, True, color), (x, y))
            y += 24

    def _frame_to_surface(self, frame: Any, pygame: Any) -> Any:
        try:
            import numpy as np

            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] < 3:
                return None
            rgb = np.ascontiguousarray(array[:, :, :3])
            height, width = rgb.shape[:2]
            return pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB").convert()
        except Exception:
            return None

    def _is_valid_frame(self, frame: Any) -> bool:
        if frame is None:
            return False
        try:
            import numpy as np

            array = np.asarray(frame)
            return array.ndim == 3 and array.shape[2] >= 3 and int(array[:, :, :3].max()) > 5
        except Exception:
            return False

    def _draw_overlay(self, screen: Any, font: Any, observation: Any, action: dict[str, float], success: bool) -> None:
        lines = [
            f"step {observation.step_id}",
            f"dist {float(observation.state.get('distance_to_target', 0.0)):.4f}",
            "W/S A/D Q/E",
            "R reset  Esc quit",
        ]
        if success:
            lines.append("success")
        y = self.panel_height + 10
        for line in lines:
            shadow = font.render(line, True, (0, 0, 0))
            text = font.render(line, True, (255, 255, 255))
            screen.blit(shadow, (11, y + 1))
            screen.blit(text, (10, y))
            y += 22

    def _draw_panel(self, screen: Any, font: Any, observation: Any, action: dict[str, float], success: bool) -> None:
        lines = [
            self.config.benchmark.name,
            f"step {observation.step_id}",
            f"success {success}",
            f"distance {float(observation.state.get('distance_to_target', 0.0)):.4f}",
            *self._panel_lines,
            "keys: W/S j0  A/D j1  Q/E j2  J/L j3  K/I j4  U/O j5  R reset  Esc quit",
            "action: " + " ".join(f"{name}={value:g}" for name, value in action.items()),
        ]
        y = 8
        for line in lines:
            screen.blit(font.render(line, True, (255, 255, 255)), (10, y))
            y += 18
            if y > self.panel_height - 18:
                break

    def _present(self, pygame: Any) -> None:
        pygame.event.pump()
        pygame.display.update()
        pygame.display.flip()
        pygame.event.pump()

    def _save_debug_screen(self, screen: Any, name: str) -> None:
        if not self.debug_frames:
            return
        try:
            import imageio.v2 as imageio
            import numpy as np

            out_dir = Path(self.config.logging.output_dir) / "debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            array = np.asarray(screen.get_view("3"), dtype="uint8").reshape((screen.get_width(), screen.get_height(), 3)).transpose(1, 0, 2)
            imageio.imwrite(out_dir / f"{name}.png", array)
        except Exception:
            pass

    def _step_once(
        self,
        *,
        sim: Any,
        task: Any,
        observation_provider: Any,
        parser: Any,
        validator: Any,
        controller: Any,
        logger: Any,
        metrics: Any,
        history: list[StepTransition],
        observation: Any,
        raw_action: dict[str, float],
        success: bool,
        running: bool,
    ) -> tuple[Any, bool, bool]:
        parsed_action = parser.parse(raw_action)
        validation = validator.validate(parsed_action)
        sim_action = controller.translate(parsed_action)
        step_result = sim.step(sim_action)
        next_observation = observation_provider.get_observation(sim)
        success = task.is_success(next_observation, sim.get_state())
        truncated = len(history) + 1 >= self.max_steps and not success
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
        if transition.terminated or transition.truncated:
            running = False
        return next_observation, success, running
