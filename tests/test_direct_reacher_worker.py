import multiprocessing as mp

import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import build_controller, build_mapping


pytest.importorskip("pygame")
pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from mllm_teleop_benchmark.ui.runtime import PlayerCommand, PlayerRuntimeResult, sim_adapter_worker_entry


def test_sim_adapter_worker_runtime_reset_step_shutdown():
    config = load_config("configs/reacher_delta_eef.yaml")
    context = mp.get_context("spawn")
    action_queue: mp.Queue = context.Queue()
    result_queue: mp.Queue = context.Queue()
    worker = context.Process(target=sim_adapter_worker_entry, args=(config, action_queue, result_queue))
    worker.start()
    try:
        action_queue.put(PlayerCommand("reset"))
        initial = result_queue.get(timeout=10)
        assert isinstance(initial, PlayerRuntimeResult)
        assert initial.observation is not None
        assert np.asarray(initial.frame).ndim == 3
        assert initial.state is not None
        assert "goal_distance" in initial.state

        controller = build_controller(config, build_mapping(config), config.benchmark.seed)
        sim_action = controller.translate(
            JoystickAction(
                {
                    "joystick_0": -1.0,
                    "joystick_1": 0.0,
                }
            )
        )
        action_queue.put(PlayerCommand("step", sim_action))
        stepped = result_queue.get(timeout=10)
        assert isinstance(stepped, PlayerRuntimeResult)
        assert stepped.observation is not None
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.state is not None
        assert "goal_distance" in stepped.state
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1

        action_queue.put(PlayerCommand("reset"))
        reset = result_queue.get(timeout=10)
        assert isinstance(reset, PlayerRuntimeResult)
        assert not reset.terminated
    finally:
        action_queue.put(PlayerCommand("close"))
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
    assert worker.exitcode == 0
