import os

import numpy as np
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.robocasa_controller import RoboCasaOSCController
from mllm_teleop_benchmark.core.registry import build_controller, build_mapping, build_sim, build_task
from mllm_teleop_benchmark.core.types import RoboCasaAction
from mllm_teleop_benchmark.mapping.frame_transform import IdentityControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.robocasa.adapter import RoboCasaSimAdapter
from mllm_teleop_benchmark.tasks.robocasa_task import RoboCasaGenericTask


CONFIG_PATH = "configs/robocasa_pick_place_counter_to_cabinet.yaml"
ROBOCASA_SETUP_HINT = (
    "Run `python3 scripts/manage_uv_env.py create robocasa`; runtime validation requires "
    "`MLLM_TELEOP_REQUIRE_ROBOCASA=1 python3 scripts/manage_uv_env.py smoke robocasa`."
)
ROBOCASA_RUNTIME_HINT = (
    "If this fails with `load_model_on_init`, the remaining blocker is the robosuite runtime "
    "source commit `824ac14cefcfb7ec125fe5eb2e0bad7364466154`, which must be fetched and "
    "installed no-deps/editable before claiming runtime verification."
)


def _require_robocasa() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        __import__("gymnasium")
        __import__("robocasa")
    except ImportError as exc:
        pytest.skip(f"robocasa is not available in this environment: {exc}. {ROBOCASA_SETUP_HINT}")
    except Exception as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_ROBOCASA") == "1":
            raise
        pytest.skip(f"robocasa failed to import in this optional-backend environment: {exc}. {ROBOCASA_SETUP_HINT}")


def _skip_unless_required(message: str, exc: Exception) -> None:
    if os.environ.get("MLLM_TELEOP_REQUIRE_ROBOCASA") == "1":
        raise exc
    pytest.skip(f"{message}: {exc}. {ROBOCASA_RUNTIME_HINT}")


def test_robocasa_config_builds_without_backend():
    config = load_config(CONFIG_PATH)

    sim = build_sim(config)
    task = build_task(config)
    mapping = build_mapping(config, seed=42)
    controller = build_controller(config, mapping, seed=42)

    assert isinstance(sim, RoboCasaSimAdapter)
    assert isinstance(task, RoboCasaGenericTask)
    assert isinstance(controller, RoboCasaOSCController)
    assert config.simulator.env_id == "robocasa/PickPlaceCounterToCabinet"
    assert config.simulator.env_kwargs["split"] == "pretrain"


def test_robocasa_controller_action_shape_contract():
    mapping = FixedJoystickMapping(
        channels={
            "joystick_0": "dx",
            "joystick_1": "dy",
            "joystick_2": "dz",
            "joystick_3": "gripper",
        },
        signs={
            "joystick_0": 1.0,
            "joystick_1": 1.0,
            "joystick_2": 1.0,
            "joystick_3": 1.0,
        },
    )
    controller = RoboCasaOSCController(
        mapping=mapping,
        frame_transform=IdentityControlFrameTransform(),
        scales={"dx": 0.1, "dy": 0.1, "dz": 0.1},
        control_mode=1.0,
    )

    action = JoystickAction(
        channels={
            "joystick_0": 1.0,
            "joystick_1": -1.0,
            "joystick_2": 0.5,
            "joystick_3": 1.0,
        }
    )
    sim_action = controller.translate(action)

    assert isinstance(sim_action, RoboCasaAction)
    assert set(sim_action.values) == {
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    }
    np.testing.assert_allclose(sim_action.values["action.end_effector_position"], [0.1, -0.1, 0.05])
    np.testing.assert_allclose(sim_action.values["action.end_effector_rotation"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(sim_action.values["action.gripper_close"], [1.0])
    np.testing.assert_allclose(sim_action.values["action.base_motion"], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(sim_action.values["action.control_mode"], [1.0])
    assert sim_action.metadata["controller"] == "robocasa_osc"


def test_robocasa_runtime_smoke_reset_step_render_close():
    _require_robocasa()
    config = load_config(CONFIG_PATH)
    adapter = RoboCasaSimAdapter(config.simulator)
    try:
        try:
            obs = adapter.reset(config.task, seed=42)
        except Exception as exc:
            _skip_unless_required("robocasa runtime or assets are unavailable", exc)

        assert obs.step_id == 0
        assert obs.images
        assert any(name in obs.images for name in ("robot0_agentview_left", "video.robot0_agentview_left"))

        mapping = build_mapping(config, seed=42)
        controller = build_controller(config, mapping, seed=42)
        sim_action = controller.translate(
            JoystickAction(
                channels={
                    "joystick_0": 0.0,
                    "joystick_1": 0.0,
                    "joystick_2": 0.0,
                    "joystick_3": 0.0,
                }
            )
        )
        step = adapter.step(sim_action)
        assert step.observation.step_id == 1

        frame = adapter.render(config.camera.cameras[0])
        assert isinstance(frame, np.ndarray)
        assert frame.shape[-1] == 3

        state = adapter.get_state()
        assert isinstance(state["is_success"], bool)
        assert "success" in step.info or "is_success" in state
    finally:
        adapter.close()
