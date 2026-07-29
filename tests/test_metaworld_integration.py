import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
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
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import MetaworldAction
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.mapping.joystick_mapping import FixedJoystickMapping
from mllm_teleop_benchmark.sim.metaworld.adapter import MetaworldSimAdapter
from mllm_teleop_benchmark.ui.runtime import MujocoPlayerRuntimeHooks, hooks_for_config


MLLM_CONFIGS = (
    (
        "configs/metaworld_reach_mllm_reflection.yaml",
        "fixed",
        "identity",
        True,
    ),
    (
        "configs/metaworld_reach_mllm_reflection_perturbed.yaml",
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)

PRIVATE_SENTINELS = {
    "private_state": "PRIVATE_METAWORLD_STATE_SENTINEL",
    "private_reward": "PRIVATE_METAWORLD_REWARD_SENTINEL",
    "private_distance": "PRIVATE_METAWORLD_DISTANCE_SENTINEL",
    "private_target_position": "PRIVATE_METAWORLD_TARGET_SENTINEL",
    "private_eef_pose": "PRIVATE_METAWORLD_EEF_POSE_SENTINEL",
    "private_simulator_action": "PRIVATE_METAWORLD_SIM_ACTION_SENTINEL",
    "private_seed": "PRIVATE_METAWORLD_SEED_SENTINEL",
}


class MetaworldFakeClient:
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
                },
                "Reflection": (
                    "The current RGB view does not yet show movement, so the "
                    "translation channels still need to be identified."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-metaworld-{len(self.requests)}",
            "model": "fake-metaworld-model",
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
    assert config.simulator.params["camera_name"] == "topview"
    assert config.simulator.params["width"] == 320
    assert config.simulator.params["height"] == 240
    assert config.task.type == "metaworld_generic"
    assert tuple(action_space.channels) == ("joystick_0", "joystick_1", "joystick_2")
    assert isinstance(sim, MetaworldSimAdapter)
    assert sim.env is None
    assert task.instruction() == "Reach the target position with the robot end-effector."
    assert provider.__class__.__name__ == "MetaworldObservationProvider"
    assert controller.metadata()["type"] == "metaworld_box"


def test_metaworld_config_uses_worker_play_runtime():
    config = load_config("configs/metaworld_reach.yaml")

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == "topview"
    assert isinstance(hooks_for_config(config), MujocoPlayerRuntimeHooks)


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

        frames = {
            camera.name: np.asarray(adapter.render(camera))
            for camera in config.camera.cameras
        }
        assert tuple(frames) == ("topview", "behindGripper")
        for pixels in frames.values():
            assert pixels.shape == (240, 320, 3)
            assert pixels.dtype == np.uint8
            assert float(pixels.mean()) > 1.0
            assert float(pixels.std()) > 1.0
        assert not np.array_equal(frames["topview"], frames["behindGripper"])
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_metaworld_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()

    assert config.simulator.params["env_name"] == "reach-v3"
    assert config.task.params["instruction"] == (
        "Move the robot end-effector to the visible target position."
    )
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz")
    assert config.control_interface.gripper == "none"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible
    assert config.control_interface.control_frame_description == (
        "the Meta-World world frame: +x points right in topview, +y points "
        "toward the far edge of the table (up in topview), and +z points "
        "upward away from the tabletop"
    )
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == ("topview", "behindGripper")
    assert config.logging.save_video is False
    assert tuple(schema) == (
        "joystick_0",
        "joystick_1",
        "joystick_2",
    )
    assert all(
        schema[f"joystick_{index}"]
        == {"kind": "discrete", "values": [-1.0, 0.0, 1.0]}
        for index in range(3)
    )

    group = make_experiment_group(config)
    assert group is not None
    assert "output-action_with_reflection" in group
    if mapping_strategy == "fixed":
        assert "__mapping-" not in group
        assert "__frame-" not in group
    else:
        assert "__mapping-permuted_sign_flipped" in group
        assert "__frame-random_rotation_3d" in group


def test_metaworld_mllm_task_instruction_is_identical_across_control_conditions():
    baseline = load_config("configs/metaworld_reach_mllm_reflection.yaml")
    perturbed = load_config(
        "configs/metaworld_reach_mllm_reflection_perturbed.yaml"
    )

    assert baseline.task.params["instruction"] == perturbed.task.params["instruction"]


def test_metaworld_mllm_prompt_describes_public_xyz_only():
    baseline = load_config("configs/metaworld_reach_mllm_reflection.yaml")
    baseline_prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text

    assert "active controllable dimensions are: dx, dy, dz" in baseline_prompt
    assert "dx, dy, and dz denote end-effector translation" in baseline_prompt
    assert "positive mapped value commands the positive direction" in baseline_prompt
    assert "joystick_0 -> dx" in baseline_prompt
    assert "joystick_1 -> dy" in baseline_prompt
    assert "joystick_2 -> dz" in baseline_prompt
    assert "+x points right in topview" in baseline_prompt
    assert "+y points toward the far edge of the table" in baseline_prompt
    assert "+z points upward away from the tabletop" in baseline_prompt
    assert "gripper" not in baseline_prompt
    assert "No additional control-frame rotation is applied" in baseline_prompt

    perturbed = load_config(
        "configs/metaworld_reach_mllm_reflection_perturbed.yaml"
    )
    metadata = control_interface_metadata(perturbed, perturbed.benchmark.seed)
    perturbed_prompt = ControlInstructionBuilder().build(
        metadata,
        build_action_space(perturbed).schema(),
    ).text

    assert "active controllable dimensions are: dx, dy, dz" in perturbed_prompt
    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert (
        "Infer the hidden controls from visual observations and your own action "
        "history"
    ) in perturbed_prompt
    assert "no worked example or sampled hidden-control value is provided" in (
        perturbed_prompt
    )
    assert "joystick_0 ->" not in perturbed_prompt
    assert "mapping" not in metadata
    assert "mapping_strategy" not in metadata
    assert "frame_transform" not in metadata
    assert "rotation_matrix" not in str(metadata)


def test_metaworld_perturbation_only_changes_translation_channels():
    config = load_config(
        "configs/metaworld_reach_mllm_reflection_perturbed.yaml"
    )
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller.frame_transform, RotationControlFrameTransform)
    assert set(mapping.channels.values()) == {"dx", "dy", "dz"}
    assert any(
        mapping.signs[f"joystick_{index}"] < 0.0 for index in range(3)
    )

    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": 0.0,
                "joystick_2": 0.0,
            }
        )
    )
    assert len(sim_action.values) == 4
    assert np.all(np.isfinite(sim_action.values))
    assert sim_action.values[3] == 1.0


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_metaworld_fake_client_closes_four_step_public_rgb_loop(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    pytest.importorskip("gymnasium")
    pytest.importorskip("metaworld")

    client = MetaworldFakeClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-metaworld-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake-metaworld.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_METAWORLD_API_KEY")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = MetaworldSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    monkeypatch.setattr(MetaworldSimAdapter, "get_state", poisoned_get_state)
    config = load_config(config_path)
    patched = replace(
        config,
        task=replace(config.task, max_steps=4, success_threshold=2.0),
        logging=replace(
            config.logging,
            output_dir=str(tmp_path),
            save_video=False,
        ),
    )

    episode = BenchmarkRunner(patched).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_image_count(request) for request in client.requests] == [2, 4, 6, 8]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    latest_request = client.requests[-1]
    assistant_messages = [
        message
        for message in latest_request.messages
        if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 3
    assert all(json.loads(message["content"])["Reflection"] for message in assistant_messages)

    request_text = _request_text(latest_request)
    assert "Camera order: topview, behindGripper" in request_text
    assert "active controllable dimensions are: dx, dy, dz" in request_text
    assert "dx, dy, and dz denote end-effector translation" in request_text
    assert "+x points right in topview" in request_text
    assert "+y points toward the far edge of the table" in request_text
    assert "+z points upward away from the tabletop" in request_text
    assert "gripper" not in request_text
    if mapping_strategy == "fixed":
        assert "joystick_0 -> dx" in request_text
        assert "No additional control-frame rotation is applied" in request_text
    else:
        assert "exact joystick-to-dimension mapping is not disclosed" in request_text
        assert "exact physical-frame transform is not disclosed" in request_text
        assert (
            "Infer the hidden controls from visual observations and your own action "
            "history"
        ) in request_text

    encoded_requests = "\n".join(
        request.encoded_body().decode("utf-8") for request in client.requests
    )
    for sentinel in (
        *PRIVATE_SENTINELS.values(),
        "PRIVATE_METAWORLD_API_KEY",
    ):
        assert sentinel not in encoded_requests
    lower_request_text = request_text.lower()
    for forbidden in (
        "native_observation",
        "reward",
        "distance_to_target",
        "ee_position",
        "simulator_action",
        "episode seed",
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
    assert all(len(step["sim_action"]["values"]) == 4 for step in trajectory)

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 8
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (240, 320, 3)
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0
    with Image.open(episode_dir / "model_calls/images/step_0000/topview.png") as image:
        topview = np.asarray(image.convert("RGB"))
    with Image.open(
        episode_dir / "model_calls/images/step_0000/behindGripper.png"
    ) as image:
        behind_gripper = np.asarray(image.convert("RGB"))
    assert not np.array_equal(topview, behind_gripper)

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v2"
    assert manifest["history"] == {
        "action_window": "all",
        "image_window": 3,
        "max_images_per_request": 128,
        "max_request_bytes": 67_108_864,
    }
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert (
        manifest["control_interface"]["mapping_visible_to_agent"] is visible
    )
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"PRIVATE_METAWORLD_API_KEY" not in artifact.read_bytes()


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
