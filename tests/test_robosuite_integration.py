import base64
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.robosuite_controller import RobosuiteOSCController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import Observation, RobosuiteAction
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import (
    IdentityControlFrameTransform,
    RotationControlFrameTransform,
)
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.robosuite_adapter import RobosuiteSimAdapter, sample_camera_randomization
from mllm_teleop_benchmark.tasks.robosuite_task import RobosuiteLiftTask


MLLM_CONFIGS = (
    (
        "configs/robosuite_lift_mllm_chat_completions.yaml",
        "fixed",
        "identity",
    ),
    (
        "configs/robosuite_lift_mllm_chat_completions_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ),
)

PRIVATE_SENTINELS = {
    "robot_proprio": "PRIVATE_STATE_SENTINEL",
    "eef_pos": "PRIVATE_EEF_POSE_SENTINEL",
    "eef_quat": "PRIVATE_EEF_QUAT_SENTINEL",
    "reward": "PRIVATE_REWARD_SENTINEL",
    "object_pose": "PRIVATE_OBJECT_POSE_SENTINEL",
    "distance": "PRIVATE_DISTANCE_SENTINEL",
    "sim_action": "PRIVATE_SIM_ACTION_SENTINEL",
    "seed": "PRIVATE_SEED_SENTINEL",
    "camera_randomization": "PRIVATE_CAMERA_RANDOMIZATION_SENTINEL",
}


class RobosuiteFakeClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    "joystick_0": 0,
                    "joystick_1": 0,
                    "joystick_2": 0,
                    "joystick_3": 0,
                },
                "Reflection": "More RGB feedback is needed to identify each motion channel.",
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-robosuite-{len(self.requests)}",
            "model": "fake-robosuite-model",
            "choices": [{"message": {"content": text}}],
        }
        return ModelResponse(
            text=text,
            request_id=raw["id"],
            model=raw["model"],
            usage=ModelUsage(input_tokens=100, output_tokens=32),
            raw=raw,
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _require_robosuite() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        __import__("robosuite")
    except ImportError as exc:
        pytest.skip(f"robosuite is not available in this environment: {exc}")
    except Exception as exc:
        if os.environ.get("MLLM_TELEOP_REQUIRE_ROBOSUITE") == "1" or "robosuite" in os.environ.get("CONDA_DEFAULT_ENV", ""):
            raise
        pytest.skip(f"robosuite failed to import in this optional-backend environment: {exc}")


def test_robosuite_config_uses_worker_play_runtime():
    config = load_config("configs/robosuite_lift.yaml")

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "agentview"


@pytest.mark.parametrize(
    "config_path",
    (
        "configs/robosuite_lift.yaml",
        "configs/robosuite_lift_mllm_chat_completions.yaml",
        "configs/robosuite_lift_mllm_chat_completions_perturbed.yaml",
    ),
)
def test_robosuite_lift_configs_preserve_upstream_task_camera_framing(config_path):
    config = load_config(config_path)

    # A position-only override keeps the old orientation and can point agentview
    # away from the robot workspace even though the resulting RGB is non-black.
    assert "camera_randomization" not in config.task.params
    assert config.task.params["camera_names"] == [
        "agentview",
        "robot0_eye_in_hand",
    ]
    assert config.task.params["camera_widths"] == [512, 256]
    assert config.task.params["camera_heights"] == [512, 256]


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform"),
    MLLM_CONFIGS,
)
def test_robosuite_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
):
    config = load_config(config_path)
    action_schema = build_action_space(config).schema()

    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.response_format == "json_object"
    assert config.agent.history.action_window == 3
    assert config.agent.history.image_window == 3
    assert config.task.success_threshold == 0.04
    assert config.task.params["reject_initial_success"] is True
    assert config.agent.observation.cameras == (
        "agentview",
        "robot0_eye_in_hand",
    )
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.gripper == "direct"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.control_frame_description == (
        "the robosuite 1.4.1 OSC MuJoCo world x, y, and z axes; "
        "translation-only commands are normalized world-frame inputs. A mapped "
        "magnitude-1 benchmark command produces +/-0.01 m translation, and in "
        "upright agentview image right is approximately +world y while image up is approximately "
        "-0.706 world x +0.708 world z. robot0_eye_in_hand is rigidly "
        "wrist-mounted and moves with the wrist, so its image-plane basis has "
        "no fixed world-axis projection. The Panda gripper treats values below "
        "0.5 as open, exactly 0.5 as hold, and values above 0.5 as closed. If "
        "a transformed translation component would exceed native magnitude 1, "
        "all three translation components are uniformly rescaled by the largest "
        "absolute component; this preserves direction but can reduce step length"
    )
    assert config.control_interface.params == {
        "scale_mode": "normalized_robosuite_1_4_1_osc_input_native_0_05m",
        "native_translation_gain_m": 0.05,
    }
    assert config.control_interface.gripper_open_command == 0.0
    assert config.control_interface.gripper_closed_command == 1.0
    assert config.control_interface.gripper_hold_command == 0.5
    assert config.controller.frame == "world_xyz"
    assert config.controller.scales == {
        "dx": 0.2,
        "dy": 0.2,
        "dz": 0.2,
        "droll": 0.0628318530717959,
        "dpitch": 0.0628318530717959,
        "dyaw": 0.0628318530717959,
        "gripper": 1.0,
    }

    agentview = next(
        camera for camera in config.camera.cameras if camera.name == "agentview"
    )
    assert agentview.params["basis_frame"] == "mujoco_world"
    assert agentview.params["basis_approximate"] is True
    image_right = np.asarray(agentview.params["image_right_in_world"])
    image_up = np.asarray(agentview.params["image_up_in_world"])
    np.testing.assert_allclose(image_right, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(image_up, [-0.706, 0.0, 0.708])
    np.testing.assert_allclose(np.linalg.norm(image_right), 1.0, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(image_up), 1.0, atol=2e-3)
    np.testing.assert_allclose(np.dot(image_right, image_up), 0.0, atol=2e-3)

    prompt = ControlInstructionBuilder().build(
        control_interface_metadata(config, config.benchmark.seed),
        action_schema,
    ).text
    assert "translation-only commands are normalized world-frame inputs" in prompt
    assert "magnitude-1 benchmark command produces +/-0.01 m" in prompt
    assert "image right is approximately +world y" in prompt
    assert "image up is approximately -0.706 world x +0.708 world z" in prompt
    assert "robot0_eye_in_hand is rigidly wrist-mounted" in prompt
    assert "has no fixed world-axis projection" in prompt
    assert "values below 0.5 as open" in prompt
    assert "exactly 0.5 as hold" in prompt
    assert "values above 0.5 as closed" in prompt
    assert tuple(action_schema) == (
        "joystick_0",
        "joystick_1",
        "joystick_2",
        "joystick_3",
    )
    assert all(
        action_schema[f"joystick_{index}"]
        == {"kind": "continuous", "min": -30.0, "max": 30.0}
        for index in range(3)
    )
    assert action_schema["joystick_3"] == {
        "kind": "continuous",
        "min": 0.0,
        "max": 1.0,
    }

    group = make_experiment_group(config)
    assert group is not None
    assert "__output-" not in group
    if mapping_strategy != "fixed":
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_robosuite_task_rejects_successful_reset_before_model_request():
    config = load_config("configs/robosuite_lift_mllm_chat_completions.yaml")
    task = RobosuiteLiftTask(config.task)

    class StubSim:
        def reset(self, task_config, seed):
            assert task_config is config.task
            assert seed == 42
            return Observation(step_id=0, images={}, state={})

        def get_state(self):
            return {"is_success": True}

    with pytest.raises(RuntimeError, match="already successful immediately after reset"):
        task.reset(StubSim(), seed=42)


def test_robosuite_perturbation_only_changes_motion_channels():
    config = load_config("configs/robosuite_lift_mllm_chat_completions_perturbed.yaml")
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller.frame_transform, RotationControlFrameTransform)
    assert set(mapping.channels.values()) == {"dx", "dy", "dz", "gripper"}
    assert mapping.channels["joystick_3"] == "gripper"
    assert mapping.signs["joystick_3"] == 1.0
    assert any(
        mapping.signs[f"joystick_{index}"] < 0.0
        for index in range(3)
    )

    action = JoystickAction(
        {
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": 0.0,
            "joystick_3": 1.0,
        }
    )
    sim_action = controller.translate(action)
    assert len(sim_action.values) == 7
    assert sim_action.values[6] == 1.0
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }
    assert controller.metadata()["translation_bound_policy"] == (
        "uniform_component_fit"
    )


def test_robosuite_hidden_rotation_preserves_all_legal_translation_directions():
    config = load_config("configs/robosuite_lift_mllm_chat_completions_perturbed.yaml")
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)
    exceeded_before_fit = 0

    for values in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        action = JoystickAction(
            {
                "joystick_0": values[0],
                "joystick_1": values[1],
                "joystick_2": values[2],
                "joystick_3": 0.0,
            }
        )
        mapped = mapping.map_action(action)
        transformed = np.asarray(
            controller.frame_transform.transform_delta_position(
                tuple(mapped.get(axis, 0.0) for axis in ("dx", "dy", "dz"))
            )
        )
        if float(np.max(np.abs(transformed))) > 1.0:
            exceeded_before_fit += 1

        actual = np.asarray(controller.translate(action).values[:3])
        assert float(np.max(np.abs(actual))) <= 1.0 + 1e-7
        if float(np.linalg.norm(transformed)) > 0.0:
            assert float(np.dot(transformed, actual)) > 0.0
            np.testing.assert_allclose(
                np.cross(transformed, actual),
                np.zeros(3),
                atol=1e-6,
            )

    assert exceeded_before_fit > 0


def test_robosuite_adapter_init():
    _require_robosuite()
    config = load_config("configs/robosuite_lift.yaml")
    adapter = RobosuiteSimAdapter(config.simulator)
    try:
        obs = adapter.reset(config.task, seed=42)

        assert "agentview" in obs.images
        assert "robot0_eye_in_hand" in obs.images
        assert obs.images["agentview"].shape == (512, 512, 3)
        assert obs.images["robot0_eye_in_hand"].shape == (256, 256, 3)
        assert all(
            image.dtype == np.uint8 and int(image.max()) > int(image.min())
            for image in obs.images.values()
        )

        state = adapter.get_state()
        assert state["step_id"] == 0
        assert "eef_pos" in state
        assert "is_success" in state

        step = adapter.step(RobosuiteAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)))
        assert step.observation.step_id == 1
        assert adapter.get_state()["step_id"] == 1

        rendered = adapter.render(config.camera.cameras[0])
        assert rendered.shape == (512, 512, 3)
        assert rendered.dtype == np.uint8
        assert int(rendered.max()) > int(rendered.min())
    finally:
        adapter.close()


def test_robosuite_experiment_seeds_reset_rgb_canary():
    _require_robosuite()
    config = load_config(MLLM_CONFIGS[0][0])
    adapter = RobosuiteSimAdapter(config.simulator)

    try:
        for seed in range(config.benchmark.seed, config.benchmark.seed + 10):
            observation = adapter.reset(config.task, seed=seed)
            assert adapter.get_state()["is_success"] is False, (
                f"robosuite seed {seed} is successful immediately after reset"
            )
            for camera in config.agent.observation.cameras:
                assert camera in observation.images, (
                    f"robosuite seed {seed} is missing public camera {camera}"
                )
                image = np.asarray(observation.images[camera])
                assert image.ndim == 3
                assert image.shape[-1] == 3
                assert image.dtype == np.uint8
                assert np.any(image != 0), (
                    f"robosuite seed {seed} camera {camera} is all black"
                )
                assert float(image.std()) > 1.0, (
                    f"robosuite seed {seed} camera {camera} has no visual variation"
                )
    finally:
        adapter.close()


def test_robosuite_render_applies_exactly_one_vertical_flip():
    config = load_config("configs/robosuite_lift.yaml")
    adapter = RobosuiteSimAdapter(config.simulator)
    raw_frame = np.array(
        [
            [[255, 0, 0], [192, 0, 0]],
            [[0, 255, 0], [0, 192, 0]],
            [[0, 0, 255], [0, 0, 192]],
        ],
        dtype=np.uint8,
    )

    class FakeSim:
        def render(self, **_kwargs):
            return raw_frame

    class FakeEnv:
        sim = FakeSim()

    adapter._env = FakeEnv()
    camera = config.camera.cameras[0]
    rendered = adapter.render(camera)
    public_observation = adapter._format_observation(
        {f"{camera.name}_image": raw_frame}
    )

    np.testing.assert_array_equal(rendered, np.flipud(raw_frame))
    np.testing.assert_array_equal(
        rendered,
        public_observation.images[camera.name],
    )
    assert not np.array_equal(rendered, raw_frame)


def test_robosuite_camera_randomization_sampling_is_seeded():
    cfg = {
        "agentview": {
            "pos_range": [[0.9, 1.1], [-0.1, 0.1], [1.4, 1.6]],
            "fovy_range": [40.0, 50.0],
        }
    }

    first = sample_camera_randomization(cfg, seed=5)
    second = sample_camera_randomization(cfg, seed=5)
    third = sample_camera_randomization(cfg, seed=6)

    assert first == second
    assert first != third
    assert len(first["agentview"]["pos"]) == 3
    assert 40.0 <= first["agentview"]["fovy"] <= 50.0


def test_robosuite_controller_translation_and_mllm_native_gain():
    mapping = FixedJoystickMapping(
        channels={"joystick_0": "dx", "joystick_1": "dy", "joystick_2": "dz", "joystick_3": "gripper"},
        signs={"joystick_0": 1.0, "joystick_1": 1.0, "joystick_2": 1.0, "joystick_3": 1.0}
    )
    transform = IdentityControlFrameTransform()
    scales = {"dx": 0.1, "dy": 0.1, "dz": 0.1, "gripper": 1.0}
    controller = RobosuiteOSCController(mapping, transform, scales)
    
    # Test movement - include all mapped channels
    action = JoystickAction(channels={"joystick_0": 1.0, "joystick_1": -1.0, "joystick_2": 0.5, "joystick_3": 0.0})
    sim_action = controller.translate(action)
    
    assert isinstance(sim_action, RobosuiteAction)
    values = np.asarray(sim_action.values)
    assert values.shape == (7,)
    # dx = 1.0 * 0.1 = 0.1
    # dy = -1.0 * 0.1 = -0.1
    # dz = 0.5 * 0.1 = 0.05
    np.testing.assert_allclose(values[:3], [0.1, -0.1, 0.05])
    
    # Test gripper: benchmark 0.0 -> robosuite -1.0
    action_open = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 0.0})
    sim_action_open = controller.translate(action_open)
    assert sim_action_open.values[6] == -1.0
    
    # Test gripper: benchmark 1.0 -> robosuite 1.0
    action_close = JoystickAction(channels={"joystick_0": 0.0, "joystick_1": 0.0, "joystick_2": 0.0, "joystick_3": 1.0})
    sim_action_close = controller.translate(action_close)
    assert sim_action_close.values[6] == 1.0

    mllm_config = load_config("configs/robosuite_lift_mllm_chat_completions.yaml")
    mllm_controller = build_controller(
        mllm_config,
        build_mapping(mllm_config, mllm_config.benchmark.seed),
        mllm_config.benchmark.seed,
    )
    normalized_action = mllm_controller.translate(
        JoystickAction(
            channels={
                "joystick_0": 1.0,
                "joystick_1": 0.0,
                "joystick_2": 0.0,
                "joystick_3": 0.5,
            }
        )
    )

    # The benchmark emits 0.2; upstream OSC_POSE maps native 1.0 to 0.05 m.
    normalized_translation = np.asarray(normalized_action.values[:3])
    np.testing.assert_allclose(normalized_translation, [0.2, 0.0, 0.0])
    np.testing.assert_allclose(
        normalized_translation
        * mllm_config.control_interface.params["native_translation_gain_m"],
        [0.01, 0.0, 0.0],
    )


def test_robosuite_pinned_osc_and_agentview_source_match_public_contract():
    robosuite = pytest.importorskip("robosuite")
    from robosuite.controllers.osc import OperationalSpaceController
    from robosuite.utils.transform_utils import quat2mat

    parameters = inspect.signature(OperationalSpaceController.__init__).parameters
    assert parameters["input_min"].default == -1
    assert parameters["input_max"].default == 1
    assert parameters["output_min"].default[:3] == (-0.05, -0.05, -0.05)
    assert parameters["output_max"].default[:3] == (0.05, 0.05, 0.05)

    arena = ET.parse(
        Path(robosuite.__file__).resolve().parent
        / "models"
        / "assets"
        / "arenas"
        / "table_arena.xml"
    ).getroot()
    camera = next(
        element
        for element in arena.iter("camera")
        if element.attrib.get("name") == "agentview"
    )
    quat_wxyz = np.fromstring(camera.attrib["quat"], sep=" ")
    rotation = quat2mat(quat_wxyz[[1, 2, 3, 0]])

    np.testing.assert_allclose(rotation[:, 0], [0.0, 1.0, 0.0], atol=2e-3)
    np.testing.assert_allclose(
        rotation[:, 1],
        [-0.706, 0.0, 0.708],
        atol=2e-3,
    )


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "_frame_transform"),
    MLLM_CONFIGS,
)
def test_robosuite_fake_client_closes_public_rgb_loop_without_private_state(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    _frame_transform,
):
    _require_robosuite()
    client = RobosuiteFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-robosuite-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_API_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = RobosuiteSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    monkeypatch.setattr(RobosuiteSimAdapter, "get_state", poisoned_get_state)
    config = load_config(
        config_path,
        (
            "task.max_steps=2",
            f"logging.output_dir={tmp_path}",
            "logging.save_video=false",
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 2
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 2
    assert [_image_count(request) for request in client.requests] == [2, 4]
    assert "Observation step 0." in _request_text(client.requests[0])
    assert "Observation step 1." in _request_text(client.requests[1])

    assistant_messages = [
        message
        for message in client.requests[1].messages
        if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 1
    prior_action = json.loads(assistant_messages[0]["content"])
    assert set(prior_action["channels"]) == {
        "joystick_0",
        "joystick_1",
        "joystick_2",
        "joystick_3",
    }
    assert prior_action["Reflection"]

    request_text = _request_text(client.requests[-1])
    assert "active controllable dimensions are: dx, dy, dz" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert '"joystick_3":{"kind":"continuous","max":1.0,"min":0.0}' in request_text
    assert "direct control" in request_text
    assert "final virtual joystick" in request_text
    assert "0 means open, 1 means closed, 0.5 means hold" in request_text
    assert "robot0_eye_in_hand is rigidly wrist-mounted" in request_text
    assert "has no fixed world-axis projection" in request_text
    assert "values below 0.5 as open" in request_text
    assert "exactly 0.5 as hold" in request_text
    assert "values above 0.5 as closed" in request_text
    assert "robosuite 1.4.1 OSC MuJoCo world x, y, and z axes" in request_text
    assert "translation-only commands are normalized world-frame inputs" in request_text
    assert "magnitude-1 benchmark command produces +/-0.01 m" in request_text
    assert "uniformly rescaled by the largest absolute component" in request_text
    assert "preserves direction but can reduce step length" in request_text
    if mapping_strategy == "fixed":
        assert "joystick_0 -> dx" in request_text
        assert "joystick_1 -> dy" in request_text
        assert "joystick_2 -> dz" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert (
            "hidden joystick mapping and control-frame transform remain fixed "
            "for the entire episode"
        ) in request_text
        assert (
            "Infer the hidden controls from visual observations and your own action "
            "history"
        ) in request_text

    for forbidden in (
        *PRIVATE_SENTINELS.values(),
        "PRIVATE_API_KEY_SENTINEL",
        "robot0_proprio-state",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "object_pose",
        "sim_action",
        "camera_randomization",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in request_text
    lower_request_text = request_text.lower()
    for forbidden_term in (
        "simulator state",
        "reward",
        "object pose",
        "eef pose",
        "end-effector pose",
        "distance",
        "sim action",
        "simulator action",
        "episode seed",
    ):
        assert forbidden_term not in lower_request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(trajectory) == 2
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 4

    assert all(
        int(np.asarray(Image.open(path).convert("RGB")).max()) > 0
        for path in pngs
    )

    request_pngs = _request_pngs(client.requests[-1])
    assert len(request_pngs) == 4
    assert all(
        image.dtype == np.uint8
        and int(image.max()) > int(image.min())
        for _, image in request_pngs
    )
    agentview_pngs = [
        image
        for camera, image in request_pngs
        if camera == "agentview"
    ]
    assert len(agentview_pngs) == 2
    for image in agentview_pngs:
        pixels = image.astype(np.int16)
        cube_mask = (
            (pixels[:, :, 0] > 100)
            & (pixels[:, :, 0] > pixels[:, :, 1] + 35)
            & (pixels[:, :, 0] > pixels[:, :, 2] + 25)
        )
        cube_rows, _ = np.nonzero(cube_mask)
        assert cube_rows.size >= 40
        assert float(np.median(cube_rows)) < image.shape[0] / 2


def _image_count(request) -> int:
    return sum(
        1
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


def _request_pngs(request) -> list[tuple[str, np.ndarray]]:
    values = []
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        camera = None
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text = item.get("text", "")
                if text.startswith("Camera ") and text.endswith(":"):
                    camera = text[len("Camera ") : -1]
                continue
            if item.get("type") != "image_url":
                continue
            assert camera is not None
            url = item["image_url"]["url"]
            prefix = "data:image/png;base64,"
            assert url.startswith(prefix)
            payload = base64.b64decode(url[len(prefix) :])
            with Image.open(io.BytesIO(payload)) as image:
                values.append((camera, np.asarray(image.convert("RGB"))))
    return values


def _request_text(request) -> str:
    values = []
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
            continue
        if not isinstance(content, list):
            continue
        values.extend(
            str(item["text"])
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return "\n".join(values)
