import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.robocasa_controller import RoboCasaOSCController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    build_sim,
    build_task,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import Observation, RoboCasaAction
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import (
    IdentityControlFrameTransform,
    RotationControlFrameTransform,
)
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.robocasa.adapter import RoboCasaSimAdapter
from mllm_teleop_benchmark.tasks.robocasa_task import RoboCasaGenericTask
from mllm_teleop_benchmark.ui.runtime import MujocoPlayerRuntimeHooks, hooks_for_config


CONFIG_PATH = "configs/robocasa_pick_place_counter_to_cabinet.yaml"
MLLM_CONFIGS = (
    (
        "configs/robocasa_pick_place_counter_to_cabinet_mllm_reflection.yaml",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/robocasa_pick_place_counter_to_cabinet_mllm_reflection_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)
PUBLIC_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
PRIVATE_SENTINELS = {
    "state": "PRIVATE_ROBOCASA_STATE_SENTINEL",
    "reward": "PRIVATE_ROBOCASA_REWARD_SENTINEL",
    "success": "PRIVATE_ROBOCASA_SUCCESS_SENTINEL",
    "distance": "PRIVATE_ROBOCASA_DISTANCE_SENTINEL",
    "object_pose": "PRIVATE_ROBOCASA_OBJECT_POSE_SENTINEL",
    "eef_pose": "PRIVATE_ROBOCASA_EEF_POSE_SENTINEL",
    "native_sim_action": "PRIVATE_ROBOCASA_NATIVE_ACTION_SENTINEL",
    "seed": "PRIVATE_ROBOCASA_SEED_SENTINEL",
    "camera_randomization": "PRIVATE_ROBOCASA_CAMERA_SENTINEL",
}
ROBOCASA_SETUP_HINT = (
    "Run `python3 scripts/manage_uv_env.py create robocasa`; runtime validation requires "
    "`MLLM_TELEOP_REQUIRE_ROBOCASA=1 python3 scripts/manage_uv_env.py smoke robocasa`."
)
ROBOCASA_RUNTIME_HINT = (
    "The current pinned RoboCasa runtime uses robosuite source commit "
    "`85abee228d1c43ab1939bce33028099945d453b4`. If source compatibility is solved but "
    "this still skips, download the upstream RoboCasa kitchen assets before claiming runtime verification."
)


class RoboCasaFakeClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        values = {
            "joystick_0": 1,
            "joystick_1": 0,
            "joystick_2": -1,
            "joystick_3": 0,
            "joystick_4": 1,
            "joystick_5": 0,
            "joystick_6": 0,
        }
        text = json.dumps(
            {
                "channels": values,
                "Reflection": (
                    "The visible motion is still being used to identify each "
                    "joystick channel."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-robocasa-{len(self.requests)}",
            "model": "fake-robocasa-model",
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
    assert config.task.params["instruction_from_observation"] is True
    assert (
        config.task.params["instruction_observation_key"]
        == "annotation.human.task_description"
    )


def test_robocasa_task_projects_only_upstream_public_instruction():
    config = load_config(CONFIG_PATH)
    task = RoboCasaGenericTask(config.task)
    public_instruction = "Pick the cucumber from the counter and place it in the cabinet."

    class StubSim:
        def reset(self, task_config, seed):
            assert task_config is config.task
            assert seed == 42
            return Observation(
                step_id=0,
                images={},
                state={
                    "annotation.human.task_description": public_instruction,
                    "state.object_pose": "PRIVATE_OBJECT_POSE",
                },
            )

    task.reset(StubSim(), seed=42)

    assert task.instruction() == public_instruction
    assert "PRIVATE_OBJECT_POSE" not in task.instruction()


def test_robocasa_task_fails_when_required_public_instruction_is_missing():
    config = load_config(CONFIG_PATH)
    task = RoboCasaGenericTask(config.task)

    class StubSim:
        def reset(self, _task_config, _seed):
            return Observation(step_id=0, images={}, state={})

    with pytest.raises(
        RuntimeError,
        match="annotation.human.task_description",
    ):
        task.reset(StubSim(), seed=42)


def test_robocasa_mllm_task_rejects_successful_reset_before_model_call():
    config = load_config(MLLM_CONFIGS[0][0])
    task = RoboCasaGenericTask(config.task)

    class StubSim:
        def reset(self, _task_config, _seed):
            return Observation(
                step_id=0,
                images={},
                state={
                    "annotation.human.task_description": (
                        "Pick the cucumber from the counter and place it in "
                        "the cabinet."
                    ),
                    "is_success": True,
                },
            )

    with pytest.raises(RuntimeError, match="already successful immediately after reset"):
        task.reset(StubSim(), seed=42)


def test_robocasa_config_uses_worker_play_runtime():
    config = load_config(CONFIG_PATH)

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "robot0_agentview_left"
    assert isinstance(hooks_for_config(config), MujocoPlayerRuntimeHooks)


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_robocasa_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    action_schema = build_action_space(config).schema()

    assert config.task.params["instruction"] == (
        "Pick the object from the counter and place it in the cabinet."
    )
    assert config.task.params["instruction_from_observation"] is True
    assert (
        config.task.params["instruction_observation_key"]
        == "annotation.human.task_description"
    )
    assert config.task.params["reject_initial_success"] is True
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.model.max_output_tokens == 192
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == PUBLIC_CAMERAS
    assert config.logging.save_video is False

    interface = config.control_interface
    assert interface.active_dimensions == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
    )
    assert interface.gripper == "binary"
    assert interface.mapping_strategy == mapping_strategy
    assert interface.mapping_visible_to_agent is visible
    assert interface.frame_transform == frame_transform
    assert interface.frame_visible_to_agent is visible
    assert interface.control_frame_description == (
        "the PandaOmron OSC robot0_right_center control frame, which is the "
        "robot-base frame and moves with the mobile base; translation and "
        "axis-angle rotation-vector commands use this moving base frame, "
        "normalized native +/-1 maps to +/-0.05 m and +/-0.5 rad with no sign "
        "compensation, and the robot0_agentview cameras are base-attached rather "
        "than fixed in world"
    )
    assert interface.params == {
        "scale_mode": "robocasa_normalized_pandaomron_osc_base_delta",
        "control_site": "robot0_right_center",
        "control_frame": "robot_base",
        "frame_moves_with_base": True,
        "native_translation_gain_m": 0.05,
        "native_rotation_gain_rad": 0.5,
        "sign_compensation": False,
    }
    assert interface.gripper_open_command == 0.0
    assert interface.gripper_hold_command is None
    assert interface.gripper_closed_command == 1.0
    assert config.controller.scales == {
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "droll": 1.0,
        "dpitch": 1.0,
        "dyaw": 1.0,
        "gripper": 1.0,
    }
    cameras = {camera.name: camera for camera in config.camera.cameras}
    for camera_name in ("robot0_agentview_left", "robot0_agentview_right"):
        assert cameras[camera_name].params == {
            "attached_to": "robot0_mobile_base",
            "fixed_world_basis": False,
        }

    prompt = ControlInstructionBuilder().build(
        control_interface_metadata(config, config.benchmark.seed),
        action_schema,
    ).text
    assert "PandaOmron OSC robot0_right_center control frame" in prompt
    assert "robot-base frame and moves with the mobile base" in prompt
    assert "axis-angle rotation-vector commands use this moving base frame" in prompt
    assert "native +/-1 maps to +/-0.05 m and +/-0.5 rad" in prompt
    assert "with no sign compensation" in prompt
    assert "robot0_agentview cameras are base-attached rather than fixed in world" in prompt

    assert tuple(action_schema) == tuple(f"joystick_{index}" for index in range(7))
    assert all(
        action_schema[f"joystick_{index}"]
        == {"kind": "discrete", "values": [-1.0, 0.0, 1.0]}
        for index in range(6)
    )
    assert action_schema["joystick_6"] == {
        "kind": "discrete",
        "values": [0.0, 1.0],
    }

    group = make_experiment_group(config)
    assert group is not None
    assert "prompt-mllm_joystick_reflection_v2" in group
    assert "output-action_with_reflection" in group
    if visible:
        assert "mapping-permuted_sign_flipped" not in group
        assert "frame-random_rotation_3d" not in group
    else:
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_robocasa_mllm_conditions_use_identical_task_text():
    instructions = {
        load_config(path).task.params["instruction"]
        for path, *_ in MLLM_CONFIGS
    }

    assert instructions == {
        "Pick the object from the counter and place it in the cabinet."
    }


def test_robocasa_perturbation_keeps_gripper_direct_and_hidden():
    config = load_config(
        "configs/robocasa_pick_place_counter_to_cabinet_mllm_reflection_perturbed.yaml"
    )
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller.frame_transform, RotationControlFrameTransform)
    assert set(mapping.channels.values()) == {
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
        "gripper",
    }
    assert mapping.channels["joystick_6"] == "gripper"
    assert mapping.signs["joystick_6"] == 1.0
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }

    action = JoystickAction(
        {
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": -1.0,
            "joystick_3": 0.0,
            "joystick_4": 1.0,
            "joystick_5": 0.0,
            "joystick_6": 0.0,
        }
    )
    command = mapping.map_action(action)
    raw_position = np.asarray(
        controller.frame_transform.transform_delta_position(
            tuple(command.get(name, 0.0) for name in ("dx", "dy", "dz"))
        )
    )
    bounded_position = controller.translate(action).values[
        "action.end_effector_position"
    ]

    assert np.max(np.abs(raw_position)) > 1.0
    assert np.max(np.abs(bounded_position)) <= 1.0
    np.testing.assert_allclose(
        np.cross(raw_position, bounded_position),
        np.zeros(3),
        atol=1e-7,
    )


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


def test_robocasa_mllm_controller_matches_official_normalized_dict_contract():
    config = load_config(
        "configs/robocasa_pick_place_counter_to_cabinet_mllm_reflection.yaml"
    )
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    sim_action = controller.translate(
        JoystickAction(
            channels={
                "joystick_0": 1.0,
                "joystick_1": -1.0,
                "joystick_2": 0.0,
                "joystick_3": -1.0,
                "joystick_4": 1.0,
                "joystick_5": 0.0,
                "joystick_6": 1.0,
            }
        )
    )

    np.testing.assert_allclose(
        sim_action.values["action.end_effector_position"],
        [1.0, -1.0, 0.0],
    )
    np.testing.assert_allclose(
        sim_action.values["action.end_effector_rotation"],
        [-1.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        sim_action.values["action.gripper_close"],
        [1.0],
    )
    np.testing.assert_allclose(
        sim_action.values["action.base_motion"],
        [0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        sim_action.values["action.control_mode"],
        [1.0],
    )
    translation_gain = config.control_interface.params["native_translation_gain_m"]
    rotation_gain = config.control_interface.params["native_rotation_gain_rad"]
    np.testing.assert_allclose(
        sim_action.values["action.end_effector_position"] * translation_gain,
        [0.05, -0.05, 0.0],
    )
    np.testing.assert_allclose(
        sim_action.values["action.end_effector_rotation"] * rotation_gain,
        [-0.5, 0.5, 0.0],
    )
    assert all(
        np.all(np.abs(np.asarray(values, dtype=float)) <= 1.0)
        for values in sim_action.values.values()
    )


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


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_robocasa_fake_client_closes_four_step_public_rgb_loop(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    _require_robocasa()
    client = RoboCasaFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-robocasa-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-robocasa.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_ROBOCASA_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = RoboCasaSimAdapter.get_state
    public_instructions = []
    original_task_reset = RoboCasaGenericTask.reset

    def inspected_task_reset(self, sim, seed):
        observation = original_task_reset(self, sim, seed)
        public_instructions.append(self.instruction())
        return observation

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    native_actions = []
    original_step = RoboCasaSimAdapter.step

    def inspected_step(self, action):
        native_action = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in action.values.items()
        }
        assert self._env.action_space.contains(native_action)
        native_actions.append(native_action)
        return original_step(self, action)

    monkeypatch.setattr(RoboCasaSimAdapter, "get_state", poisoned_get_state)
    monkeypatch.setattr(RoboCasaSimAdapter, "step", inspected_step)
    monkeypatch.setattr(RoboCasaGenericTask, "reset", inspected_task_reset)
    config = load_config(
        config_path,
        (
            "task.max_steps=4",
            f"logging.output_dir={tmp_path}",
            "logging.save_video=false",
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert len(native_actions) == 4
    assert len(public_instructions) == 1
    assert public_instructions[0]
    assert [_image_count(request) for request in client.requests] == [3, 6, 9, 12]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    assistant_history = [
        json.loads(message["content"])
        for message in client.requests[-1].messages
        if message["role"] == "assistant"
    ]
    assert len(assistant_history) == 3
    assert all(item["Reflection"] for item in assistant_history)
    assert all(
        set(item["channels"]) == {
            f"joystick_{index}" for index in range(7)
        }
        for item in assistant_history
    )

    request_text = _request_text(client.requests[-1])
    assert "Prompt version: mllm_joystick_reflection_v2" in request_text
    assert f"Task: {public_instructions[0]}" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "droll, dpitch, and dyaw denote end-effector rotation" in request_text
    assert "PandaOmron OSC robot0_right_center control frame" in request_text
    assert "robot-base frame and moves with the mobile base" in request_text
    assert "axis-angle rotation-vector commands use this moving base frame" in request_text
    assert "native +/-1 maps to +/-0.05 m and +/-0.5 rad" in request_text
    assert "with no sign compensation" in request_text
    assert "base-attached rather than fixed in world" in request_text
    assert "binary control" in request_text
    assert "0 means open, 1 means closed" in request_text
    assert "means hold" not in request_text
    if visible:
        assert "joystick_0 -> dx" in request_text
        assert "joystick_5 -> dyaw" in request_text
        assert "joystick_6 -> gripper" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert (
            "hidden joystick mapping and control-frame transform remain fixed "
            "for the entire episode"
        ) in request_text
        assert (
            "Infer the hidden controls from visual observations and your own "
            "action history"
        ) in request_text

    encoded_requests = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    for sentinel in (*PRIVATE_SENTINELS.values(), "PRIVATE_ROBOCASA_API_KEY"):
        assert sentinel not in encoded_requests
    lower_request_text = request_text.lower()
    for forbidden in (
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "reward",
        "success oracle",
        "object pose",
        "eef pose",
        "native sim action",
        "simulator action",
        "episode seed",
        "camera randomization",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in lower_request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)
    for step in trajectory:
        values = step["sim_action"]["values"]
        assert set(values) == {
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        }
        assert np.asarray(values["action.end_effector_position"]).shape == (3,)
        assert np.asarray(values["action.end_effector_rotation"]).shape == (3,)
        assert np.asarray(values["action.gripper_close"]).shape == (1,)
        assert np.asarray(values["action.base_motion"]).shape == (4,)
        assert np.asarray(values["action.control_mode"]).shape == (1,)
        assert all(
            np.all(np.isfinite(np.asarray(native_values, dtype=float)))
            and np.all(np.abs(np.asarray(native_values, dtype=float)) <= 1.0)
            for native_values in values.values()
        )

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 12
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (256, 256, 3)
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v2"
    assert manifest["observation"]["cameras"] == list(PUBLIC_CAMERAS)
    assert manifest["history"] == {
        "action_window": "all",
        "image_window": 3,
        "max_images_per_request": 128,
        "max_request_bytes": 67_108_864,
    }
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert manifest["control_interface"]["mapping_visible_to_agent"] is visible
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"PRIVATE_ROBOCASA_API_KEY" not in artifact.read_bytes()


def _image_count(request) -> int:
    return sum(
        1
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


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
