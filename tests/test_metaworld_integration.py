import numpy as np
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.metaworld import MetaworldBoxActionController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    build_task,
)
from mllm_teleop_benchmark.core.types import MetaworldAction
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.metaworld.adapter import MetaworldSimAdapter


def test_metaworld_reach_config_builds_without_metaworld_install():
    config = load_config("configs/metaworld_reach.yaml")
    action_space = build_action_space(config)
    sim = build_sim(config)
    task = build_task(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    provider = build_observation_provider(config, camera_manager)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)

    assert config.simulator.type == "metaworld"
    assert config.simulator.params["env_name"] == "reach-v3"
    assert config.task.type == "metaworld_generic"
    assert tuple(action_space.channels) == ("joystick_0", "joystick_1", "joystick_2")
    assert isinstance(sim, MetaworldSimAdapter)
    assert sim.env is None
    assert task.instruction() == "Reach the target position with the robot end-effector."
    assert provider.__class__.__name__ == "MetaworldObservationProvider"
    assert controller.metadata()["type"] == "metaworld_box"


def test_metaworld_box_controller_maps_xyz_and_default_gripper():
    config = load_config("configs/metaworld_reach.yaml")
    controller = MetaworldBoxActionController.from_config(config.controller, build_mapping(config, config.benchmark.seed))

    sim_action = controller.translate(JoystickAction({"joystick_0": 1.0, "joystick_1": -1.0, "joystick_2": 0.0}))

    assert isinstance(sim_action, MetaworldAction)
    assert sim_action.values == (1.0, -1.0, 0.0, 1.0)
    assert sim_action.metadata["controller"] == "metaworld_box"


def test_metaworld_box_controller_applies_3d_frame_rotation():
    config = load_config("configs/metaworld_reach.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = MetaworldBoxActionController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(JoystickAction({"joystick_0": 1.0, "joystick_1": 0.0, "joystick_2": 0.0}))

    np.testing.assert_allclose(sim_action.values, (0.0, 1.0, 0.0, 1.0), atol=1e-12)


def test_metaworld_box_controller_maps_direct_gripper_to_native_range():
    mapping = FixedJoystickMapping({"joystick_0": "gripper"})
    controller = MetaworldBoxActionController(
        mapping=mapping,
        dimensions=("gripper",),
        scales={},
        gripper_open_value=-1.0,
        gripper_closed_value=1.0,
    )

    assert controller.translate(JoystickAction({"joystick_0": 0.0})).values == (-1.0,)
    assert controller.translate(JoystickAction({"joystick_0": 1.0})).values == (1.0,)


def test_metaworld_adapter_clips_native_action_against_box_space():
    config = load_config("configs/metaworld_reach.yaml")
    adapter = MetaworldSimAdapter(config.simulator)
    adapter.env = type(
        "FakeEnv",
        (),
        {
            "action_space": type(
                "FakeBox",
                (),
                {"shape": (4,), "low": np.asarray([-1.0, -1.0, -1.0, -1.0]), "high": np.asarray([1.0, 1.0, 1.0, 1.0])},
            )()
        },
    )()

    native, metadata = adapter._native_action(MetaworldAction(values=(2.0, -2.0, 0.25, 1.5)))

    assert native.tolist() == [1.0, -1.0, 0.25, 1.0]
    assert metadata["clipped"] is True


def test_metaworld_runtime_reset_step_render_close_when_installed():
    pytest.importorskip("gymnasium")
    pytest.importorskip("metaworld")

    config = load_config("configs/metaworld_reach.yaml")
    adapter = MetaworldSimAdapter(config.simulator)
    try:
        observation = adapter.reset(config.task, seed=config.benchmark.seed)
        assert observation.step_id == 0
        assert observation.metadata["simulator"] == "metaworld"

        controller = build_controller(config, build_mapping(config, config.benchmark.seed), config.benchmark.seed)
        sim_action = controller.translate(JoystickAction({"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0}))
        step = adapter.step(sim_action)
        assert step.info["step_id"] == 1

        frame = adapter.render(config.camera.cameras[0])
        assert np.asarray(frame).ndim == 3
    finally:
        adapter.close()
