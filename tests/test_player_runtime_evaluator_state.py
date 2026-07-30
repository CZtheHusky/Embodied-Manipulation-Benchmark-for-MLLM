from __future__ import annotations

import queue
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core import registry
from mllm_teleop_benchmark.core.types import Observation, SimulatorAction, StepResult
from mllm_teleop_benchmark.ui import human_player as human_player_module
from mllm_teleop_benchmark.ui.human_player import PygameHumanPlayer
from mllm_teleop_benchmark.ui.runtime import PlayerCommand, PlayerRuntimeResult, sim_adapter_worker_entry


def test_worker_keeps_evaluator_state_separate_from_public_observation(monkeypatch):
    config = load_config("configs/dummy_reach.yaml")

    class FakeSim:
        def __init__(self):
            self.step_id = -1
            self.state_reads = []
            self.closed = False

        def reset(self, _task_config, _seed):
            self.step_id = 0

        def step(self, _action):
            self.step_id = 1
            return StepResult(
                observation=public_observation(self.step_id),
                terminated=False,
                truncated=False,
                info={"public_backend": "fake"},
            )

        def get_state(self):
            self.state_reads.append(self.step_id)
            return {
                "privileged_success": self.step_id == 1,
                "privileged_score": 100 + self.step_id,
            }

        def close(self):
            self.closed = True

    class FakeTask:
        def reset(self, sim, seed):
            sim.reset(config.task, seed)

    class FakeCameraManager:
        def reset(self, _seed):
            return None

    class FakeObservationProvider:
        def configure_sim(self, _sim):
            return None

        def get_observation(self, sim):
            return public_observation(sim.step_id)

    def public_observation(step_id):
        return Observation(
            step_id=step_id,
            images={"front": f"frame-{step_id}"},
            state={"public_step": step_id},
        )

    sim = FakeSim()
    monkeypatch.setattr(registry, "build_sim", lambda _config: sim)
    monkeypatch.setattr(registry, "build_task", lambda _config: FakeTask())
    monkeypatch.setattr(registry, "build_camera_manager", lambda *_args: FakeCameraManager())
    monkeypatch.setattr(registry, "build_observation_provider", lambda *_args: FakeObservationProvider())

    command_queue = queue.Queue()
    result_queue = queue.Queue()
    command_queue.put(PlayerCommand("reset"))
    command_queue.put(PlayerCommand("step", SimulatorAction()))
    command_queue.put(PlayerCommand("close"))

    sim_adapter_worker_entry(config, command_queue, result_queue)

    initial = result_queue.get_nowait()
    stepped = result_queue.get_nowait()

    assert initial.state == {"public_step": 0}
    assert initial.observation is not None
    assert initial.observation.state == {"public_step": 0}
    assert initial.evaluator_state == {
        "privileged_success": False,
        "privileged_score": 100,
    }
    assert "privileged_success" not in initial.observation.state

    assert stepped.state == {"public_step": 1}
    assert stepped.observation is not None
    assert stepped.observation.state == {"public_step": 1}
    assert stepped.evaluator_state == {
        "privileged_success": True,
        "privileged_score": 101,
    }
    assert "privileged_success" not in stepped.observation.state
    assert stepped.step_result is not None
    assert stepped.step_result.info == {
        "public_backend": "fake",
        "frame_name": "front",
    }
    assert sim.state_reads == [0, 1]
    assert sim.closed


def test_human_player_uses_evaluator_state_for_success_and_final_metrics(monkeypatch, tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    config = replace(
        config,
        logging=replace(
            config.logging,
            output_dir=tmp_path,
            save_images=False,
            save_trajectory=False,
            save_config_snapshot=False,
        ),
        ui=replace(config.ui, max_steps=5),
    )

    initial_observation = Observation(
        step_id=0,
        images={"front": "initial-frame"},
        state={"step_id": 0, "public_status": "running"},
    )
    stepped_observation = Observation(
        step_id=1,
        images={"front": "stepped-frame"},
        state={"step_id": 1, "public_status": "running"},
    )
    final_evaluator_state = {
        "privileged_success": True,
        "privileged_score": 37,
    }

    class FakeRuntime:
        def __init__(self):
            self.started = False
            self.closed = False

        def start(self):
            self.started = True

        def reset(self):
            return PlayerRuntimeResult(
                frame="initial-frame",
                observation=initial_observation,
                state=dict(initial_observation.state),
                step_result=None,
                evaluator_state={
                    "privileged_success": False,
                    "privileged_score": 0,
                },
            )

        def step(self, _action):
            return PlayerRuntimeResult(
                frame="stepped-frame",
                observation=stepped_observation,
                state=dict(stepped_observation.state),
                step_result=StepResult(
                    observation=stepped_observation,
                    terminated=False,
                    truncated=False,
                    info={"public_backend": "fake"},
                ),
                evaluator_state=final_evaluator_state,
            )

        def close(self):
            self.closed = True

    class FakeTask:
        def __init__(self):
            self.success_calls: list[tuple[Observation, dict[str, Any]]] = []
            self.metrics_calls: list[dict[str, Any]] = []

        def is_success(self, observation, evaluator_state):
            self.success_calls.append((observation, evaluator_state))
            return bool(evaluator_state["privileged_success"])

        def metrics(self, evaluator_state):
            self.metrics_calls.append(evaluator_state)
            return {"privileged_score": evaluator_state["privileged_score"]}

    class FakeClock:
        def tick(self, _fps):
            return None

    key_event = SimpleNamespace(type=2, key=ord("w"))
    pygame = SimpleNamespace(
        QUIT=1,
        KEYDOWN=2,
        K_ESCAPE=27,
        K_p=ord("p"),
        K_r=ord("r"),
        init=lambda: None,
        quit=lambda: None,
        event=SimpleNamespace(get=lambda: [key_event]),
        key=SimpleNamespace(name=lambda _key: "w"),
        font=SimpleNamespace(Font=lambda *_args: object()),
        time=SimpleNamespace(Clock=FakeClock),
        display=SimpleNamespace(
            set_mode=lambda _size: object(),
            set_caption=lambda _caption: None,
            flip=lambda: None,
        ),
    )

    runtime = FakeRuntime()
    task = FakeTask()
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setattr(human_player_module, "build_task", lambda _config: task)

    player = PygameHumanPlayer(config, runtime)
    monkeypatch.setattr(player, "_configure_canvas_from_frame", lambda _frame: None)
    monkeypatch.setattr(player, "_draw", lambda *_args, **_kwargs: None)

    result = player.run()

    assert runtime.started
    assert runtime.closed
    assert result.success
    assert result.steps == 1
    assert result.termination_reason == "success"
    assert result.metrics["privileged_score"] == 37
    assert task.success_calls == [(stepped_observation, final_evaluator_state)]
    assert task.metrics_calls == [final_evaluator_state]
    assert "privileged_success" not in stepped_observation.state
