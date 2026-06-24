from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    build_task,
)
from mllm_teleop_benchmark.core.types import LiberoAction
from mllm_teleop_benchmark.sim.libero.adapter import LiberoSimAdapter
from mllm_teleop_benchmark.ui.runtime import MujocoPlayerRuntimeHooks, hooks_for_config


class _FakeCameraManager:
    def __init__(self, cameras=()) -> None:
        self._cameras = tuple(cameras)

    def get_active_cameras(self):
        return self._cameras

    def metadata(self):
        return {"manager": "fixed"}


def _import_runtime_dependency(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_LIBERO") == "1":
            raise
        pytest.skip(f"LIBERO runtime dependency unavailable: {module_name}: {exc}")


def test_libero_config_builds_without_libero_install():
    config = load_config("configs/libero_spatial_task0.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)
    sim = build_sim(config)
    task = build_task(config)
    provider = build_observation_provider(config, camera_manager=_FakeCameraManager())

    assert config.simulator.type == "libero"
    assert config.task.type == "libero_generic"
    assert tuple(action_space.channels) == tuple(f"joystick_{idx}" for idx in range(7))
    assert controller.metadata()["type"] == "libero_delta_eef"
    assert isinstance(sim, LiberoSimAdapter)
    assert task.instruction() == "pick up the black bowl between the plate and the ramekin and place it on the plate"
    assert provider.__class__.__name__ == "LiberoObservationProvider"


def test_libero_config_uses_worker_play_runtime():
    config = load_config("configs/libero_spatial_task0.yaml")

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "agentview"
    assert isinstance(hooks_for_config(config), MujocoPlayerRuntimeHooks)


def test_libero_delta_eef_controller_outputs_7d_action():
    config = load_config("configs/libero_spatial_task0.yaml")
    controller = build_controller(config, build_mapping(config, config.benchmark.seed), config.benchmark.seed)
    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": -1.0,
                "joystick_2": 0.5,
                "joystick_3": 1.0,
                "joystick_4": 0.0,
                "joystick_5": -1.0,
                "joystick_6": 1.0,
            }
        )
    )

    assert isinstance(sim_action, LiberoAction)
    assert len(sim_action.values) == 7
    np.testing.assert_allclose(sim_action.values, [0.05, -0.05, 0.025, 0.1, 0.0, -0.1, 1.0])


def test_libero_adapter_extracts_rgb_images_and_validity_metadata():
    config = load_config("configs/libero_spatial_task0.yaml")
    adapter = LiberoSimAdapter(config.simulator)
    adapter.last_native_obs = {
        "agentview_image": np.ones((1, 4, 5, 4), dtype=np.float32),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
    }
    adapter._task_metadata = {
        "language_instruction": config.task.params["language_instruction"],
        "suite_name": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "no_language_baseline": False,
        "random_baseline": True,
    }

    images = adapter._extract_images(adapter.last_native_obs)
    state = adapter.get_state()

    assert images["agentview"].shape == (4, 5, 3)
    assert images["agentview"].dtype == np.uint8
    assert state["ee_position"] == [0.1, 0.2, 0.3]
    assert state["benchmark_validity"]["suite_name"] == "libero_spatial"
    assert state["benchmark_validity"]["random_baseline"] is True


def test_libero_runtime_reset_init_state_step_render_smoke_when_installed():
    _import_runtime_dependency("libero")
    _import_runtime_dependency("robosuite")

    config = load_config("configs/libero_spatial_task0.yaml")
    sim = build_sim(config)
    task = build_task(config)
    provider = build_observation_provider(config, camera_manager=_FakeCameraManager(config.camera.cameras))
    provider.configure_sim(sim)
    try:
        try:
            obs = task.reset(sim, seed=7)
            result = sim.step(LiberoAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)))
            observation = provider.get_observation(sim)
            frame = observation.images.get("agentview")
            if frame is None:
                frame = sim.render(config.camera.cameras[0])
        except (ImportError, FileNotFoundError, RuntimeError, AssertionError) as exc:
            if os.environ.get("MLLM_TELEOP_REQUIRE_LIBERO") == "1":
                raise
            pytest.skip(f"LIBERO runtime unavailable in this optional-backend environment: {exc}")

        assert obs.step_id == 0
        assert sim.get_state()["benchmark_validity"]["init_state_id"] == 0
        assert result.observation.step_id == 1
        assert np.asarray(frame).shape[-1] == 3
    finally:
        sim.close()
