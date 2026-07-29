import base64
from dataclasses import replace
from io import BytesIO
import multiprocessing as mp

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.agents.model_policy import MLLMPolicy
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import FrameTransformConfig
from mllm_teleop_benchmark.control.maniskill import ManiSkillDeltaEEFController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import (
    ManiSkillAction,
    Observation,
    StepTransition,
    ValidationResult,
)
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.sim.maniskill.adapter import ManiSkillSimAdapter
from mllm_teleop_benchmark.sim.maniskill.camera import maniskill_sensor_config
from mllm_teleop_benchmark.sim.maniskill.observation import ManiSkillObservationProvider


MLLM_CONFIGS = (
    "configs/maniskill_pickcube_delta_eef_mllm_fixed.yaml",
    "configs/maniskill_pickcube_delta_eef_mllm_perturbed.yaml",
)


class RecordingManiSkillClient:
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        channels = ",".join(f'"joystick_{index}":0' for index in range(7))
        text = (
            '{"channels":{'
            + channels
            + '},"Reflection":"No visible motion has been observed yet."}'
        )
        return ModelResponse(
            text=text,
            request_id=f"fake-{len(self.requests)}",
            model="fake-maniskill-model",
            usage=ModelUsage(input_tokens=100, output_tokens=16, cached_input_tokens=0),
            raw={"id": f"fake-{len(self.requests)}", "model": "fake-maniskill-model"},
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def test_maniskill_pickcube_config_builds_without_maniskill_install():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    action_space = build_action_space(config)
    controller = build_controller(config, build_mapping(config, seed=7), seed=7)

    assert config.simulator.type == "maniskill"
    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert tuple(action_space.channels) == tuple(f"joystick_{idx}" for idx in range(7))
    assert controller.metadata()["type"] == "maniskill_delta_eef"


@pytest.mark.parametrize(
    ("path", "mapping_strategy", "frame_transform"),
    (
        (MLLM_CONFIGS[0], "fixed", "identity"),
        (
            MLLM_CONFIGS[1],
            "permuted_sign_flipped",
            "random_rotation_3d",
        ),
    ),
)
def test_maniskill_pickcube_mllm_configs_define_reflection_and_public_controls(
    path,
    mapping_strategy,
    frame_transform,
):
    config = load_config(path)
    action_space = build_action_space(config)
    mapping_config = config.control_interface.to_mapping_config(seed=config.benchmark.seed)
    controller = build_controller(
        config,
        build_mapping(config, seed=config.benchmark.seed),
        seed=config.benchmark.seed,
    )

    assert config.agent.type == "mllm"
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.response_format == "json_object"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.image_window == 3
    assert config.agent.history.action_window == "all"
    assert config.agent.observation.cameras == ("render_camera",)
    assert config.camera.manager == "fixed"
    assert config.camera.cameras[0].params["source"] == "human_render"
    assert config.camera.visible_metadata is False
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.control_frame_description == (
        "the ManiSkill end-effector delta-pose controller x, y, and z axes"
    )
    assert config.control_interface.gripper_open_command == 0.0
    assert config.control_interface.gripper_closed_command == 1.0
    assert tuple(action_space.channels) == tuple(f"joystick_{index}" for index in range(7))
    assert action_space.schema()["joystick_6"] == {
        "kind": "continuous",
        "min": 0.0,
        "max": 1.0,
    }
    assert mapping_config.channels["joystick_6"] == "gripper"
    assert mapping_config.signs["joystick_6"] == 1.0
    assert controller.metadata()["frame_transform"]["type"] == frame_transform

    if mapping_strategy == "fixed":
        assert tuple(mapping_config.channels.values()) == (
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
            "gripper",
        )
        assert config.control_interface.mapping_visible_to_agent is True
        assert config.control_interface.frame_visible_to_agent is True
    else:
        assert set(tuple(mapping_config.channels.values())[:-1]) == {
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
        }
        assert tuple(mapping_config.channels.values())[:-1] != config.control_interface.active_dimensions
        assert any(value < 0.0 for key, value in mapping_config.signs.items() if key != "joystick_6")
        assert config.control_interface.mapping_visible_to_agent is False
        assert config.control_interface.frame_visible_to_agent is False
        assert controller.metadata()["frame_transform"] == {
            "type": "random_rotation_3d",
            "visible_to_agent": False,
        }


@pytest.mark.parametrize("path", MLLM_CONFIGS)
def test_maniskill_mllm_request_exposes_public_rules_without_privileged_inputs(path):
    config = load_config(path)
    client = RecordingManiSkillClient()
    policy = MLLMPolicy(config.agent, client, "fake-maniskill-model")
    action_space = build_action_space(config)
    previous = Observation(
        step_id=0,
        images={"render_camera": np.full((12, 16, 3), 64, dtype=np.uint8)},
        state={
            "state": "PRIVATE_STATE",
            "reward": "PRIVATE_REWARD",
            "cube_pose": "PRIVATE_CUBE_POSE",
            "goal_pose": "PRIVATE_GOAL_POSE",
            "eef_pose": "PRIVATE_EEF_POSE",
            "distance": "PRIVATE_DISTANCE",
            "seed": "PRIVATE_SEED",
        },
        metadata={"camera_randomization": "PRIVATE_CAMERA_RANDOMIZATION"},
    )
    current = Observation(
        step_id=1,
        images={"render_camera": np.full((12, 16, 3), 128, dtype=np.uint8)},
        state={"state": "PRIVATE_NEXT_STATE"},
        metadata={"camera_randomization": "PRIVATE_NEXT_CAMERA_RANDOMIZATION"},
    )
    parsed_action = {f"joystick_{index}": 0.0 for index in range(7)}
    transition = StepTransition(
        step_id=0,
        observation=previous,
        raw_action="PRIVATE_RAW_MODEL_OUTPUT",
        parsed_action=parsed_action,
        validation=ValidationResult(True),
        sim_action=ManiSkillAction(
            values=(0.0,) * 7,
            metadata={"sim_action": "PRIVATE_SIM_ACTION"},
        ),
        next_observation=current,
        terminated=False,
        truncated=False,
        info={"reward": "PRIVATE_STEP_REWARD", "seed": "PRIVATE_STEP_SEED"},
        reflection="No visible motion has been observed yet.",
    )
    public_control = control_interface_metadata(config, seed=config.benchmark.seed)
    public_control["private_mapping_sample"] = "PRIVATE_MAPPING_SAMPLE"
    public_control["private_frame_sample"] = "PRIVATE_FRAME_SAMPLE"
    context = AgentContext(
        observation=current,
        task_instruction=str(config.task.params["instruction"]),
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={
            "control_interface": public_control,
            "controller": {"private": "PRIVATE_CONTROLLER"},
        },
        history=(transition,),
    )

    decision = policy.act(context)

    assert decision.trace.status == "ok"
    request_text = client.requests[0].encoded_body().decode("utf-8")
    assert "Each motion joystick value must be one of -1, 0, or 1" in request_text
    assert (
        "dx, dy, and dz denote end-effector translation along the control frame's "
        "x, y, and z axes"
    ) in request_text
    assert "gripper is always the final virtual joystick" in request_text
    assert "0 means open, 1 means closed" in request_text
    assert "ManiSkill end-effector delta-pose controller" in request_text
    assert request_text.count('"role":"assistant"') == 1
    for forbidden in (
        "PRIVATE_STATE",
        "PRIVATE_REWARD",
        "PRIVATE_CUBE_POSE",
        "PRIVATE_GOAL_POSE",
        "PRIVATE_EEF_POSE",
        "PRIVATE_DISTANCE",
        "PRIVATE_SEED",
        "PRIVATE_CAMERA_RANDOMIZATION",
        "PRIVATE_NEXT_STATE",
        "PRIVATE_NEXT_CAMERA_RANDOMIZATION",
        "PRIVATE_RAW_MODEL_OUTPUT",
        "PRIVATE_SIM_ACTION",
        "PRIVATE_STEP_REWARD",
        "PRIVATE_STEP_SEED",
        "PRIVATE_MAPPING_SAMPLE",
        "PRIVATE_FRAME_SAMPLE",
        "PRIVATE_CONTROLLER",
    ):
        assert forbidden not in request_text

    if config.control_interface.mapping_strategy == "fixed":
        assert "joystick_0 -> dx" in request_text
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
        assert "joystick_0 ->" not in request_text
        assert "rotation_matrix" not in request_text
        assert "quaternion" not in request_text


def test_pygame_maniskill_player_builds_without_starting_env():
    from mllm_teleop_benchmark.ui.human_player import PygameHumanPlayer
    from mllm_teleop_benchmark.ui.registry import build_player

    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    player = build_player(config)

    assert isinstance(player, PygameHumanPlayer)
    assert player.max_steps == 100


def test_maniskill_delta_eef_controller_maps_pose_and_gripper():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    controller = ManiSkillDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed))
    action = JoystickAction(
        {
            "joystick_0": 1,
            "joystick_1": -1,
            "joystick_2": 0,
            "joystick_3": 1,
            "joystick_4": 0,
            "joystick_5": -1,
            "joystick_6": 1.0,
        }
    )

    sim_action = controller.translate(action)

    assert isinstance(sim_action, ManiSkillAction)
    assert sim_action.values == (1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0)


def test_maniskill_delta_eef_controller_applies_frame_rotation():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = ManiSkillDeltaEEFController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1,
                "joystick_1": 0,
                "joystick_2": 0,
                "joystick_3": 0,
                "joystick_4": 0,
                "joystick_5": 0,
                "joystick_6": 0.0,
            }
        )
    )

    assert abs(sim_action.values[0]) < 1e-9
    assert abs(sim_action.values[1] - 1.0) < 1e-9
    assert abs(sim_action.values[2]) < 1e-9


def test_maniskill_human_render_camera_is_not_forwarded_as_sensor_config():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    manager = build_camera_manager(config.camera, seed=5)
    camera = manager.get_active_cameras()[0]

    assert camera.name == "render_camera"
    assert camera.params["source"] == "human_render"
    assert maniskill_sensor_config(manager.get_active_cameras()) == {}


def test_maniskill_observation_provider_extracts_rgb_images():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    provider = ManiSkillObservationProvider(build_camera_manager(config.camera, seed=5))
    frame = np.zeros((1, 4, 5, 4), dtype=np.uint8)
    frame[..., 1] = 255

    images = provider._extract_images({"sensor_data": {"base_camera": {"rgb": frame}}})

    assert images["base_camera"].shape == (4, 5, 3)
    assert images["base_camera"].dtype == np.uint8
    assert images["base_camera"][0, 0, 1] == 255


def test_maniskill_adapter_clips_native_action_against_box_space():
    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    adapter = ManiSkillSimAdapter(config.simulator)
    adapter.env = type(
        "FakeEnv",
        (),
        {
            "action_space": type(
                "FakeBox",
                (),
                {"shape": (3,), "low": np.asarray([-1.0, -0.5, -1.0]), "high": np.asarray([1.0, 0.5, 1.0])},
            )()
        },
    )()

    native, metadata = adapter._native_action(ManiSkillAction(values=(2.0, -1.0, 0.25)))

    assert native.tolist() == [1.0, -0.5, 0.25]
    assert metadata["clipped"] is True


@pytest.mark.parametrize("path", MLLM_CONFIGS)
def test_maniskill_pickcube_cpu_rgb_and_fake_client_smoke(
    monkeypatch,
    tmp_path,
    path,
):
    pytest.importorskip("mani_skill")
    config = load_config(path)
    config = replace(
        config,
        task=replace(config.task, max_steps=5),
        logging=replace(config.logging, output_dir=str(tmp_path)),
    )
    client = RecordingManiSkillClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-maniskill-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-maniskill.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    episode = BenchmarkRunner(config).run().episodes[0]

    assert episode.steps == 5
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 5
    assert [
        sum(
            1
            for message in request.messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if isinstance(item, dict) and item.get("type") == "image_url"
        )
        for request in client.requests
    ] == [1, 2, 3, 4, 4]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3, 4]
    image_urls = [
        item["image_url"]["url"]
        for message in client.requests[0].messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    ]
    assert len(image_urls) == 1
    image_bytes = base64.b64decode(image_urls[0].split(",", 1)[1])
    rgb = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
    assert rgb.shape == (512, 512, 3)
    assert rgb.dtype == np.uint8
    assert np.any(rgb != 0)
    red_object_pixels = (
        (rgb[:, :, 0] > 140)
        & (rgb[:, :, 1] < 80)
        & (rgb[:, :, 2] < 80)
    )
    green_goal_pixels = (
        (rgb[:, :, 1] > 70)
        & (rgb[:, :, 0] < 80)
        & (rgb[:, :, 2] < 80)
    )
    assert np.count_nonzero(red_object_pixels) > 100
    assert np.count_nonzero(green_goal_pixels) > 5
    request_text = "\n".join(
        request.encoded_body().decode("utf-8")
        for request in client.requests
    )
    for forbidden in (
        "cube_pose",
        "goal_pose",
        "eef_pose",
        "tcp_pose",
        "distance",
        "reward",
        "sim_action",
        "seed",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in request_text


def test_sim_adapter_worker_maniskill_reset_step_shutdown_when_installed():
    pytest.importorskip("mani_skill")
    from mllm_teleop_benchmark.ui.runtime import PlayerCommand, PlayerRuntimeResult, sim_adapter_worker_entry

    config = load_config("configs/maniskill_pickcube_delta_eef.yaml")
    context = mp.get_context("spawn")
    action_queue: mp.Queue = context.Queue()
    result_queue: mp.Queue = context.Queue()
    worker = context.Process(target=sim_adapter_worker_entry, args=(config, action_queue, result_queue))
    worker.start()
    try:
        action_queue.put(PlayerCommand("reset"))
        initial = result_queue.get(timeout=30)
        assert isinstance(initial, PlayerRuntimeResult)
        if initial.error:
            pytest.skip(f"ManiSkill runtime unavailable: {initial.error}")
        assert initial.observation is not None
        assert np.asarray(initial.frame).ndim == 3

        controller = build_controller(config, build_mapping(config), config.benchmark.seed)
        sim_action = controller.translate(
            JoystickAction(
                {
                    "joystick_0": 0.0,
                    "joystick_1": 0.0,
                    "joystick_2": 0.0,
                    "joystick_3": 0.0,
                    "joystick_4": 0.0,
                    "joystick_5": 0.0,
                    "joystick_6": 0.0,
                }
            )
        )
        action_queue.put(PlayerCommand("step", sim_action))
        stepped = result_queue.get(timeout=30)
        assert isinstance(stepped, PlayerRuntimeResult)
        if stepped.error:
            pytest.skip(f"ManiSkill runtime step unavailable: {stepped.error}")
        assert stepped.observation is not None
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1
    finally:
        action_queue.put(PlayerCommand("close"))
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
