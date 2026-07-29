from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.schema import TaskConfig
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_frame_transform,
    build_mapping,
    build_sim,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import (
    DMControlAction,
    DMControlPlanarEEFAction,
)
from mllm_teleop_benchmark.ui.runtime import (
    PlayerCommand,
    PlayerRuntimeResult,
    sim_adapter_worker_entry,
)

BASELINE_MLLM_CONFIG = "configs/dmcontrol_manipulator_bring_ball_mllm.yaml"
PERTURBED_MLLM_CONFIG = "configs/dmcontrol_manipulator_bring_ball_mllm_perturbed.yaml"


class FakeDMControlClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        actions = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (-1.0, 0.0, 0.5),
            (0.0, -1.0, 0.5),
        )
        dx, dy, gripper = actions[(len(self.requests) - 1) % len(actions)]
        text = json.dumps(
            {
                "channels": {
                    "joystick_0": dx,
                    "joystick_1": dy,
                    "joystick_2": gripper,
                },
                "Reflection": "The previous public images inform the next planar control action.",
            }
        )
        return ModelResponse(
            text=text,
            request_id=f"fake-dmcontrol-{len(self.requests)}",
            model="fake-dmcontrol-model",
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=12,
                cached_input_tokens=20,
            ),
            raw={
                "id": f"fake-dmcontrol-{len(self.requests)}",
                "model": "fake-dmcontrol-model",
                "choices": [{"message": {"content": text}}],
            },
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


def _request_image_count(request) -> int:
    return sum(
        1
        for message in request.messages
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if isinstance(item, dict) and item.get("type") == "image_url"
    )


def _request_text(request) -> str:
    parts = []
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(item["text"])
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return "\n".join(parts)


def test_dmcontrol_short_episode(tmp_path):
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config("configs/dmcontrol_example.yaml")
    config = replace(
        config,
        benchmark=replace(config.benchmark, name="test_dmcontrol", num_episodes=1),
        task=replace(config.task, max_steps=2),
        logging=replace(config.logging, output_dir=str(tmp_path), save_images=False),
    )

    result = BenchmarkRunner(config).run()

    assert result.aggregate_metrics["num_episodes"] == 1
    assert 1 <= result.episodes[0].steps <= 2
    assert "reward" in result.episodes[0].metrics
    assert "success" in result.episodes[0].metrics


def test_dmcontrol_manipulator_native_action_is_actuator_space():
    suite = pytest.importorskip("dm_control.suite")
    env = suite.load("manipulator", "bring_ball")

    spec = env.action_spec()
    actuator_names = [
        env.physics.model.id2name(index, "actuator")
        for index in range(env.physics.model.nu)
    ]

    assert spec.shape == (5,)
    assert actuator_names == ["root", "shoulder", "elbow", "wrist", "grasp"]
    assert actuator_names != ["dx", "dy", "dz", "droll", "gripper"]


def test_dmcontrol_legacy_controller_translation_is_shape_compatible():
    config = load_config("configs/dmcontrol_example.yaml")
    controller = build_controller(config, build_mapping(config))
    action = JoystickAction(
        channels={
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": -1.0,
            "joystick_3": 0.5,
            "joystick_4": 1.0,
        }
    )

    sim_action = controller.translate(action)

    assert isinstance(sim_action, DMControlAction)
    assert sim_action.values == (0.1, 0.0, -0.1, 0.05, 1.0)


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    [
        (BASELINE_MLLM_CONFIG, "fixed", "identity", True),
        (
            PERTURBED_MLLM_CONFIG,
            "permuted_sign_flipped",
            "random_rotation_2d",
            False,
        ),
    ],
)
def test_dmcontrol_mllm_config_is_planar_eef_not_native_motor_aliases(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()

    assert config.controller.type == "dmcontrol_planar_eef"
    assert config.controller.params["dimensions"] == ["dx", "dy", "gripper"]
    assert config.control_interface.mode == "planar_eef_delta_gripper"
    assert config.control_interface.active_dimensions == ("dx", "dy")
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible
    assert config.control_interface.gripper_open_command == 0.0
    assert config.control_interface.gripper_hold_command == 0.5
    assert config.control_interface.gripper_closed_command == 1.0
    assert config.control_interface.params["scale_mode"] == (
        "dmcontrol_world_xz_cartesian_impedance_target_0_03m"
    )
    assert config.controller.scales == {
        "dx": 0.03,
        "dy": 0.03,
        "gripper": 1.0,
    }
    assert tuple(schema) == ("joystick_0", "joystick_1", "joystick_2")
    assert schema["joystick_0"]["values"] == [-1.0, 0.0, 1.0]
    assert schema["joystick_1"]["values"] == [-1.0, 0.0, 1.0]
    assert schema["joystick_2"] == {
        "kind": "continuous",
        "min": 0.0,
        "max": 1.0,
    }
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.image_window == 3
    assert config.agent.history.action_window == "all"


def test_dmcontrol_planar_controller_applies_mapping_and_rotation():
    baseline = load_config(BASELINE_MLLM_CONFIG)
    baseline_action = build_controller(
        baseline,
        build_mapping(baseline, baseline.benchmark.seed),
        baseline.benchmark.seed,
    ).translate(
        JoystickAction(
            {"joystick_0": 1.0, "joystick_1": 0.0, "joystick_2": 0.25}
        )
    )
    assert isinstance(baseline_action, DMControlPlanarEEFAction)
    np.testing.assert_allclose(baseline_action.delta_xy, (0.03, 0.0), atol=1e-12)
    assert baseline_action.gripper == 0.25

    perturbed = load_config(PERTURBED_MLLM_CONFIG)
    seed = perturbed.benchmark.seed
    mapping = build_mapping(perturbed, seed)
    transform = build_frame_transform(perturbed, seed)
    joystick = JoystickAction(
        {"joystick_0": 1.0, "joystick_1": 0.0, "joystick_2": 0.25}
    )
    mapped = mapping.map_action(joystick)
    expected = transform.transform_delta_position(
        (mapped.get("dx") * 0.03, mapped.get("dy") * 0.03, 0.0)
    )
    perturbed_action = build_controller(
        perturbed,
        mapping,
        seed,
    ).translate(joystick)

    assert isinstance(perturbed_action, DMControlPlanarEEFAction)
    assert mapping.channels == {
        "joystick_0": "dy",
        "joystick_1": "dx",
        "joystick_2": "gripper",
    }
    assert mapping.signs["joystick_0"] == -1.0
    np.testing.assert_allclose(perturbed_action.delta_xy, expected[:2], atol=1e-12)
    assert not np.allclose(perturbed_action.delta_xy, baseline_action.delta_xy)
    assert perturbed_action.metadata["frame_transform"] == {
        "type": "random_rotation_2d",
        "visible_to_agent": False,
    }


def test_dmcontrol_mllm_prompt_describes_true_public_control_contract():
    baseline = load_config(BASELINE_MLLM_CONFIG)
    baseline_prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text

    assert "active controllable dimensions are: dx, dy" in baseline_prompt
    assert "only dx and dy are active and there is no dz or rotation command" in baseline_prompt
    assert "control dy is world z" in baseline_prompt
    assert "mapped magnitude-1 command requests a 0.03 m" in baseline_prompt
    assert "hand camera tracks the moving hand" in baseline_prompt
    assert "must not be applied to the hand view" in baseline_prompt
    assert "joystick_0 -> dx" in baseline_prompt
    assert "joystick_1 -> dy" in baseline_prompt
    assert "joystick_2 -> gripper" in baseline_prompt
    assert "0 means open" in baseline_prompt
    assert "0.5 means hold" in baseline_prompt
    assert "1 means closed" in baseline_prompt
    for false_dimension in ("dz", "droll", "dpitch", "dyaw"):
        assert f"joystick_2 -> {false_dimension}" not in baseline_prompt

    perturbed = load_config(PERTURBED_MLLM_CONFIG)
    perturbed_metadata = control_interface_metadata(
        perturbed,
        perturbed.benchmark.seed,
    )
    perturbed_prompt = ControlInstructionBuilder().build(
        perturbed_metadata,
        build_action_space(perturbed).schema(),
    ).text

    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert "remain fixed for the entire episode" in perturbed_prompt
    assert "magnitude 1 requests a 0.03 m" in perturbed_prompt
    assert "hand camera tracks the moving hand" in perturbed_prompt
    assert "joystick_0 ->" not in perturbed_prompt
    assert "mapping" not in perturbed_metadata
    assert "rotation_matrix" not in str(perturbed_metadata)


def test_dmcontrol_pinned_camera_xml_matches_public_fixed_view_contract():
    dm_control = pytest.importorskip("dm_control")
    xml_root = ET.parse(
        Path(dm_control.__file__).resolve().parent / "suite" / "manipulator.xml"
    ).getroot()
    cameras = {
        camera.attrib["name"]: camera
        for camera in xml_root.iter("camera")
        if "name" in camera.attrib
    }

    assert cameras["fixed"].attrib["xyaxes"] == "1 0 0 0 0 1"
    assert cameras["fixed"].attrib.get("mode") is None
    assert cameras["hand"].attrib["xyaxes"] == "1 0 0 0 0 1"
    assert cameras["hand"].attrib["mode"] == "track"
    hand_body = next(
        body for body in xml_root.iter("body") if body.attrib.get("name") == "hand"
    )
    assert hand_body.find(".//camera[@name='hand']") is cameras["hand"]


@pytest.mark.parametrize(
    ("channel", "value", "site_axis"),
    [
        ("joystick_0", 1.0, 0),
        ("joystick_0", -1.0, 0),
        ("joystick_1", 1.0, 1),
        ("joystick_1", -1.0, 1),
    ],
)
def test_dmcontrol_planar_eef_single_axis_moves_grasp_in_expected_world_direction(
    channel,
    value,
    site_axis,
):
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config(BASELINE_MLLM_CONFIG)
    sim = build_sim(config)
    try:
        sim.reset(config.task, config.benchmark.seed)
        controller = build_controller(
            config,
            build_mapping(config, config.benchmark.seed),
            config.benchmark.seed,
        )
        joystick = {
            "joystick_0": 0.0,
            "joystick_1": 0.0,
            "joystick_2": 0.5,
        }
        joystick[channel] = value
        before = np.asarray(
            sim.env.physics.named.data.site_xpos["grasp", ["x", "z"]],
            dtype=np.float64,
        ).copy()

        result = sim.step(controller.translate(JoystickAction(joystick)))

        after = np.asarray(
            sim.env.physics.named.data.site_xpos["grasp", ["x", "z"]],
            dtype=np.float64,
        ).copy()
        signed_motion = (after[site_axis] - before[site_axis]) * np.sign(value)
        assert signed_motion > 5e-4
        assert result.info["step_id"] == 1
    finally:
        sim.close()


def test_dmcontrol_perturbed_mapping_and_rotation_reach_real_physics():
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config(PERTURBED_MLLM_CONFIG)
    sim = build_sim(config)
    try:
        sim.reset(config.task, config.benchmark.seed)
        controller = build_controller(
            config,
            build_mapping(config, config.benchmark.seed),
            config.benchmark.seed,
        )
        action = controller.translate(
            JoystickAction(
                {
                    "joystick_0": 1.0,
                    "joystick_1": 0.0,
                    "joystick_2": 0.5,
                }
            )
        )
        before = np.asarray(
            sim.env.physics.named.data.site_xpos["grasp", ["x", "z"]],
            dtype=np.float64,
        ).copy()

        sim.step(action)

        motion = (
            np.asarray(
                sim.env.physics.named.data.site_xpos["grasp", ["x", "z"]],
                dtype=np.float64,
            )
            - before
        )
        command = np.asarray(action.delta_xy)
        cosine = float(
            np.dot(command, motion)
            / (np.linalg.norm(command) * np.linalg.norm(motion))
        )
        assert cosine > 0.9
    finally:
        sim.close()


def test_dmcontrol_planar_eef_gripper_zero_opens_and_one_closes():
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config(BASELINE_MLLM_CONFIG)
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    apertures = {}

    for gripper in (0.0, 1.0):
        sim = build_sim(config)
        try:
            sim.reset(config.task, config.benchmark.seed)
            action = controller.translate(
                JoystickAction(
                    {
                        "joystick_0": 0.0,
                        "joystick_1": 0.0,
                        "joystick_2": gripper,
                    }
                )
            )
            for _ in range(20):
                sim.step(action)
            physics = sim.env.physics
            apertures[gripper] = abs(
                float(
                    physics.named.data.site_xpos["thumbtip_touch", "x"]
                    - physics.named.data.site_xpos["fingertip_touch", "x"]
                )
            )
        finally:
            sim.close()

    assert apertures[0.0] > apertures[1.0] + 0.04


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    [
        (BASELINE_MLLM_CONFIG, "fixed", "identity", True),
        (
            PERTURBED_MLLM_CONFIG,
            "permuted_sign_flipped",
            "random_rotation_2d",
            False,
        ),
    ],
)
def test_dmcontrol_mllm_fake_client_preserves_history_rgb_and_privacy(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config(config_path)
    config = replace(
        config,
        task=replace(config.task, max_steps=4, success_threshold=2.0),
        logging=replace(config.logging, output_dir=str(tmp_path)),
    )
    client = FakeDMControlClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-dmcontrol-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-dmcontrol.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_FAKE_DMCONTROL_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    episode = BenchmarkRunner(config).run().episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_request_image_count(request) for request in client.requests] == [
        2,
        4,
        6,
        8,
    ]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    request_text = "\n".join(_request_text(request) for request in client.requests)
    request_text_lower = request_text.lower()
    for forbidden in (
        "observation/",
        "arm_pos",
        "arm_vel",
        "touch",
        "hand_pos",
        "object_pos",
        "object_vel",
        "target_pos",
        "reward",
        "step_type",
        "distance",
        "sim_action",
        "seed",
        "jacobian",
        "qpos",
        "qvel",
        "root",
        "shoulder",
        "elbow",
        "wrist",
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_2d",
    ):
        assert forbidden not in request_text_lower

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(
        set(step["sim_action"]) >= {"delta_xy", "gripper", "metadata"}
        for step in trajectory
    )
    assert all(len(step["sim_action"]["delta_xy"]) == 2 for step in trajectory)
    assert [step["reflection"] for step in trajectory] == [
        "The previous public images inform the next planar control action."
    ] * 4

    image_paths = sorted((episode_dir / "model_calls/images").glob("step_*/*.png"))
    assert len(image_paths) == 8
    for image_path in image_paths:
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (240, 320, 3)
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    run_dir = episode_dir.parent
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["history"]["image_window"] == 3
    assert manifest["history"]["action_window"] == "all"
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert manifest["control_interface"]["mapping_visible_to_agent"] is visible
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible
    if mapping_strategy == "fixed":
        assert "__mapping-" not in run_dir.parent.name
        assert "__frame-" not in run_dir.parent.name
    else:
        assert "__mapping-permuted_sign_flipped" in run_dir.parent.name
        assert "__frame-random_rotation_2d" in run_dir.parent.name

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"PRIVATE_FAKE_DMCONTROL_KEY" not in artifact.read_bytes()


def test_dmcontrol_reward_success_key_uses_threshold():
    from mllm_teleop_benchmark.tasks.dmcontrol_task import DMControlTask

    task = DMControlTask(
        TaskConfig(
            type="dmcontrol_generic",
            max_steps=10,
            success_threshold=0.95,
            params={"success_key": "reward"},
        )
    )

    assert task.is_success(None, {"reward": 0.5}) is False
    assert task.is_success(None, {"reward": 0.95}) is True


def test_dmcontrol_action_to_native_clips_and_validates_shape():
    pytest.importorskip("dm_control")
    from dm_env import specs

    from mllm_teleop_benchmark.sim.dmcontrol.actions import action_to_native

    action_spec = specs.BoundedArray(
        shape=(2,),
        dtype=np.float64,
        minimum=[-1.0, -0.5],
        maximum=[1.0, 0.5],
    )

    clipped = action_to_native(DMControlAction(values=(2.0, -2.0)), action_spec)
    np.testing.assert_allclose(clipped, np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="shape"):
        action_to_native(DMControlAction(values=(0.0, 0.0, 0.0)), action_spec)


def test_dmcontrol_named_camera_config_applies_to_model():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import (
        DMControlObservationProvider,
    )

    class FakeModel:
        def __init__(self):
            self.cam_pos = np.zeros((1, 3), dtype=np.float64)
            self.cam_quat = np.zeros((1, 4), dtype=np.float64)
            self.cam_fovy = np.zeros((1,), dtype=np.float64)

        def name2id(self, name, kind):
            assert name == "fixed"
            assert kind == "camera"
            return 0

    class FakePhysics:
        def __init__(self):
            self.model = FakeModel()

    physics = FakePhysics()
    provider = DMControlObservationProvider(camera_manager=None)
    provider._apply_mujoco_camera_config(
        physics,
        {
            "name": "fixed",
            "pose": (1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5),
            "fovy": 45.0,
        },
    )

    np.testing.assert_allclose(
        physics.model.cam_pos[0],
        np.array([1.0, 2.0, 3.0]),
    )
    np.testing.assert_allclose(
        physics.model.cam_quat[0],
        np.array([0.5, 0.5, 0.5, 0.5]),
    )
    assert physics.model.cam_fovy[0] == 45.0


def test_dmcontrol_free_camera_randomization_fails_loudly():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import (
        DMControlObservationProvider,
    )

    provider = DMControlObservationProvider(camera_manager=None)
    with pytest.raises(NotImplementedError, match="free-camera orbit"):
        provider._apply_mujoco_camera_config(
            object(),
            {
                "distance": 1.0,
                "azimuth": 10.0,
                "elevation": -30.0,
                "lookat": (0.0, 0.0, 0.0),
            },
        )


def test_dmcontrol_sim_adapter_worker_reset_step_shutdown():
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")
    config = load_config("configs/dmcontrol_example.yaml")
    context = mp.get_context("spawn")
    action_queue: mp.Queue = context.Queue()
    result_queue: mp.Queue = context.Queue()
    worker = context.Process(
        target=sim_adapter_worker_entry,
        args=(config, action_queue, result_queue),
    )
    worker.start()
    try:
        action_queue.put(PlayerCommand("reset"))
        initial = result_queue.get(timeout=15)
        assert isinstance(initial, PlayerRuntimeResult)
        assert initial.error is None
        assert initial.observation is not None
        assert np.asarray(initial.frame).ndim == 3

        controller = build_controller(
            config,
            build_mapping(config),
            config.benchmark.seed,
        )
        sim_action = controller.translate(
            JoystickAction(
                {
                    f"joystick_{index}": 0.0
                    for index in range(5)
                }
            )
        )
        action_queue.put(PlayerCommand("step", sim_action))
        stepped = result_queue.get(timeout=15)
        assert isinstance(stepped, PlayerRuntimeResult)
        assert stepped.error is None
        assert stepped.observation is not None
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1
    finally:
        action_queue.put(PlayerCommand("close"))
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
