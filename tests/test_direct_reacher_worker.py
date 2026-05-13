import multiprocessing as mp

import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import build_controller, build_mapping
from mllm_teleop_benchmark.ui.direct_gymnasium_player import WorkerCommand, WorkerResult, env_worker


pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")


def test_direct_gymnasium_worker_reset_step_shutdown():
    config = load_config("configs/reacher_delta_eef.yaml")
    action_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    worker = mp.Process(target=env_worker, args=(config, action_queue, result_queue))
    worker.start()
    try:
        initial = result_queue.get(timeout=10)
        assert isinstance(initial, WorkerResult)
        assert np.asarray(initial.frame).ndim == 3
        assert "goal_distance" in initial.info

        controller = build_controller(config, build_mapping(config), config.benchmark.seed)
        sim_action = controller.translate(
            JoystickAction(
                {
                    "joystick_0": -1.0,
                    "joystick_1": 0.0,
                }
            )
        )
        action_queue.put(WorkerCommand("step", sim_action))
        stepped = result_queue.get(timeout=10)
        assert isinstance(stepped, WorkerResult)
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert "goal_distance" in stepped.info
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1

        action_queue.put(WorkerCommand("reset"))
        reset = result_queue.get(timeout=10)
        assert isinstance(reset, WorkerResult)
        assert not reset.done
    finally:
        action_queue.put(None)
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
    assert worker.exitcode == 0
