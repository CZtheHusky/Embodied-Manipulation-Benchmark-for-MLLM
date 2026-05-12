from __future__ import annotations

import os
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import EpisodeResult


@dataclass(frozen=True)
class WorkerCommand:
    kind: str
    action: Any = None


@dataclass(frozen=True)
class WorkerResult:
    frame: Any
    info: dict[str, Any]
    done: bool


def env_worker(config: BenchmarkConfig, action_queue: mp.Queue, res_queue: mp.Queue) -> None:
    """Isolated process for MuJoCo to avoid OpenGL context conflicts."""
    # Use EGL in its own process
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    
    import gymnasium as gym
    from mllm_teleop_benchmark.sim.gymnasium_envs.registry import (
        REGISTERED_REACHER_DELTA_EEF_ID,
        register_gymnasium_envs,
    )
    
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
    
    _, info = env.reset(seed=config.benchmark.seed)
    # Send first frame
    res_queue.put(WorkerResult(env.render(), dict(info), False))
    
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

            if kind == "reset":
                _, info = env.reset(seed=config.benchmark.seed)
                res_queue.put(WorkerResult(env.render(), dict(info), False))
            elif kind == "step":
                _, _, terminated, truncated, info = env.step(action)
                res_queue.put(WorkerResult(env.render(), dict(info), terminated or truncated))
            else:
                raise ValueError(f"Unknown worker command kind: {kind}")
    finally:
        env.close()

class DirectReacherPlayer:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        ui_config = config.raw.get("ui", {}) if isinstance(config.raw.get("ui", {}), dict) else {}
        self.width = int(ui_config.get("window_width", 640))
        self.height = int(ui_config.get("window_height", 480))
        self.max_steps = int(ui_config.get("max_steps", 500))

    def run(self) -> EpisodeResult:
        # Use simple software-backed Pygame in the main process
        os.environ["SDL_VIDEODRIVER"] = "x11"
        
        action_queue = mp.Queue()
        res_queue = mp.Queue()
        
        worker = mp.Process(target=env_worker, args=(self.config, action_queue, res_queue))
        worker.start()
        
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        font = pygame.font.Font(None, 24)
        
        actions = {
            pygame.K_w: np.array([-1.0, 0.0], dtype=np.float32),
            pygame.K_s: np.array([1.0, 0.0], dtype=np.float32),
            pygame.K_a: np.array([0.0, -1.0], dtype=np.float32),
            pygame.K_d: np.array([0.0, 1.0], dtype=np.float32),
            pygame.K_q: np.array([-0.707, -0.707], dtype=np.float32),
            pygame.K_e: np.array([-0.707, 0.707], dtype=np.float32),
            pygame.K_z: np.array([0.707, -0.707], dtype=np.float32),
            pygame.K_x: np.array([0.707, 0.707], dtype=np.float32),
        }

        steps = 0
        success = False
        last_info: dict[str, Any] = {}
        
        try:
            # Get initial frame
            initial = res_queue.get()
            frame, last_info, _ = initial.frame, initial.info, initial.done
            self._draw(screen, font, frame, 0, None)
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
                            frame, last_info, _ = result.frame, result.info, result.done
                            steps = 0
                            self._draw(screen, font, frame, steps, None)
                            pygame.display.flip()
                        elif event.key in actions:
                            action_queue.put(WorkerCommand("step", actions[event.key]))
                            result = res_queue.get()
                            frame, last_info, done = result.frame, result.info, result.done
                            steps += 1
                            dist = last_info.get("goal_distance")
                            success = dist is not None and float(dist) <= self.config.task.success_threshold
                            self._draw(screen, font, frame, steps, dist)
                            pygame.display.flip()
                            if done:
                                running = False
                time.sleep(0.01)
        finally:
            action_queue.put(None)
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
            pygame.quit()

        metrics = {
            "success": success,
            "steps": steps,
            "invalid_action_count": 0,
            "final_distance_to_target": float(last_info.get("goal_distance", float("inf"))),
            "goal_hold_counter": int(last_info.get("goal_hold_counter", 0)),
        }
        return EpisodeResult(
            episode_id=0,
            seed=self.config.benchmark.seed,
            success=success,
            steps=steps,
            metrics=metrics,
            log_dir=str(Path(self.config.logging.output_dir) / "episode_0000"),
        )

    def _draw(self, screen, font, frame, steps, distance):
        arr = np.asarray(frame)
        if arr.ndim == 3:
            if arr.shape[-1] == 4: arr = arr[..., :3]
        else:
            arr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        surface = pygame.image.frombuffer(arr.tobytes(), (arr.shape[1], arr.shape[0]), "RGB").convert()
        if surface.get_size() != (self.width, self.height):
            surface = pygame.transform.scale(surface, (self.width, self.height))
        screen.blit(surface, (0, 0))

        # Restore multi-line hints
        lines = [f"step {steps}", "W/S A/D Q/E/Z/X", "R reset  Esc quit"]
        if distance is not None:
            lines.insert(1, f"dist {float(distance):.4f}")
        
        y = 10
        for line in lines:
            screen.blit(font.render(line, True, (255, 255, 255)), (10, y))
            y += 22
