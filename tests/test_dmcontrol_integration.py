import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.schema import FrameTransformConfig, TaskConfig
from mllm_teleop_benchmark.control.dmcontrol_controller import DMControlController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import DMControlAction
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.ui.runtime import PlayerCommand, PlayerRuntimeResult, sim_adapter_worker_entry


BASELINE_MLLM_CONFIG = "configs/dmcontrol_manipulator_bring_ball_mllm.yaml"
PERTURBED_MLLM_CONFIG = "configs/dmcontrol_manipulator_bring_ball_mllm_perturbed.yaml"


class FakeDMControlClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {f"joystick_{index}": 0 for index in range(5)},
                "Reflection": "No directional effect observed from the zero action.",
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

    # Use the example config but override for a very short run
    config = load_config("configs/dmcontrol_example.yaml")

    # Simple patching of the config for testing
    patched_logging = config.logging.__class__(
        output_dir=str(tmp_path),
        save_images=False,
        save_trajectory=True,
    )
    patched_benchmark = config.benchmark.__class__(
        name="test_dmcontrol",
        num_episodes=1,
        seed=42,
    )
    patched_task = config.task.__class__(
        type=config.task.type,
        max_steps=2,
        success_threshold=config.task.success_threshold,
        params=config.task.params,
    )

    patched = config.__class__(
        **{
            **config.__dict__,
            "logging": patched_logging,
            "benchmark": patched_benchmark,
            "task": patched_task,
        }
    )

    runner = BenchmarkRunner(patched)
    result = runner.run()

    assert result.aggregate_metrics["num_episodes"] == 1
    assert 1 <= result.episodes[0].steps <= 2
    assert "reward" in result.episodes[0].metrics
    assert "success" in result.episodes[0].metrics


def test_dmcontrol_controller_translation():
    config = load_config("configs/dmcontrol_example.yaml")
    mapping = build_mapping(config)
    controller = build_controller(config, mapping)

    # Test mapping a simple joystick action
    action = JoystickAction(
        channels={
            "joystick_0": 1.0,
            "joystick_1": 0.0,
            "joystick_2": -1.0,
            "joystick_3": 0.5,
            "joystick_4": 1.0,
        }
    )  # dx, dy, dz, droll, gripper
    sim_action = controller.translate(action)

    assert isinstance(sim_action, DMControlAction)
    # Check scales application: dx: 0.1, dy: 0.1, dz: 0.1, droll: 0.1, gripper: 1.0
    assert sim_action.values == (0.1, 0.0, -0.1, 0.05, 1.0)


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "mapping_visible", "frame_visible"),
    [
        (BASELINE_MLLM_CONFIG, "fixed", "identity", True, True),
        (
            PERTURBED_MLLM_CONFIG,
            "permuted_sign_flipped",
            "random_rotation_3d",
            False,
            False,
        ),
    ],
)
def test_dmcontrol_mllm_config_contract(
    config_path,
    mapping_strategy,
    frame_transform,
    mapping_visible,
    frame_visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()

    assert config.simulator.params["domain_name"] == "manipulator"
    assert config.simulator.params["task_name"] == "bring_ball"
    assert config.controller.params["dimensions"] == ["dx", "dy", "dz", "droll", "gripper"]
    assert config.control_interface.active_dimensions == ("dx", "dy", "dz", "droll")
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is mapping_visible
    assert config.control_interface.frame_visible_to_agent is frame_visible
    assert config.agent.action_output_mode == "action_with_reflection"
    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.agent.observation.cameras == ("fixed", "hand")
    assert tuple(schema) == tuple(f"joystick_{index}" for index in range(5))
    assert all(schema[f"joystick_{index}"]["values"] == [-1.0, 0.0, 1.0] for index in range(4))
    assert schema["joystick_4"] == {"kind": "continuous", "min": 0.0, "max": 1.0}

    controller = build_controller(config, build_mapping(config, config.benchmark.seed), config.benchmark.seed)
    sim_action = controller.translate(
        JoystickAction({f"joystick_{index}": 1.0 for index in range(5)})
    )
    assert len(sim_action.values) == 5
    assert np.all(np.isfinite(sim_action.values))


def test_dmcontrol_mllm_public_prompt_discloses_only_allowed_control_rules():
    baseline = load_config(BASELINE_MLLM_CONFIG)
    baseline_schema = build_action_space(baseline).schema()
    baseline_metadata = control_interface_metadata(baseline, baseline.benchmark.seed)
    baseline_prompt = ControlInstructionBuilder().build(baseline_metadata, baseline_schema).text

    assert "Each motion joystick value must be one of -1, 0, or 1" in baseline_prompt
    assert "active controllable dimensions are: dx, dy, dz, droll" in baseline_prompt
    assert "joystick_0 -> dx" in baseline_prompt
    assert "joystick_3 -> droll" in baseline_prompt
    assert "gripper is always the final virtual joystick" in baseline_prompt

    perturbed = load_config(PERTURBED_MLLM_CONFIG)
    perturbed_schema = build_action_space(perturbed).schema()
    perturbed_metadata = control_interface_metadata(perturbed, perturbed.benchmark.seed)
    perturbed_prompt = ControlInstructionBuilder().build(perturbed_metadata, perturbed_schema).text

    assert "Each motion joystick value must be one of -1, 0, or 1" in perturbed_prompt
    assert "active controllable dimensions are: dx, dy, dz, droll" in perturbed_prompt
    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert "joystick_0 ->" not in perturbed_prompt
    assert "mapping" not in perturbed_metadata
    assert "mapping_strategy" not in perturbed_metadata
    assert "frame_transform" not in perturbed_metadata
    assert "rotation_matrix" not in str(perturbed_metadata)


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    [
        (BASELINE_MLLM_CONFIG, "fixed", "identity", True),
        (
            PERTURBED_MLLM_CONFIG,
            "permuted_sign_flipped",
            "random_rotation_3d",
            False,
        ),
    ],
)
def test_dmcontrol_mllm_fake_client_integration(
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
    patched = replace(
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

    episode = BenchmarkRunner(patched).run().episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_request_image_count(request) for request in client.requests] == [2, 4, 6, 8]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    request_text = "\n".join(_request_text(request) for request in client.requests).lower()
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
        "rotation_matrix",
        "quaternion_xyzw",
        "permuted_sign_flipped",
        "random_rotation_3d",
    ):
        assert forbidden not in request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(len(step["sim_action"]["values"]) == 5 for step in trajectory)

    for camera in ("fixed", "hand"):
        with Image.open(episode_dir / f"model_calls/images/step_0000/{camera}.png") as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.shape == (240, 320, 3)
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    run_dir = episode_dir.parent
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["action_output_mode"] == "action_with_reflection"
    assert manifest["model"]["prompt_version"] == "mllm_joystick_reflection_v1"
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
    if mapping_strategy == "fixed":
        assert "__mapping-" not in run_dir.parent.name
        assert "__frame-" not in run_dir.parent.name
    else:
        assert "__mapping-permuted_sign_flipped" in run_dir.parent.name
        assert "__frame-random_rotation_3d" in run_dir.parent.name

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            assert b"PRIVATE_FAKE_DMCONTROL_KEY" not in artifact.read_bytes()


def test_dmcontrol_controller_applies_frame_rotation_to_position_and_rotation():
    config = load_config("configs/dmcontrol_example.yaml")
    transform = RotationControlFrameTransform.from_config(
        FrameTransformConfig(type="rotation_3d", visible_to_agent=True, params={"rpy_degrees": [0, 0, 90]})
    )
    controller = DMControlController.from_config(config.controller, build_mapping(config, config.benchmark.seed), transform)

    sim_action = controller.translate(
        JoystickAction(
            {
                "joystick_0": 1.0,
                "joystick_1": 0.0,
                "joystick_2": 0.0,
                "joystick_3": 1.0,
                "joystick_4": 0.0,
            }
        )
    )

    np.testing.assert_allclose(sim_action.values[:3], [0.0, 0.1, 0.0], atol=1e-12)
    assert abs(sim_action.values[3]) < 1e-12


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

    action_spec = specs.BoundedArray(shape=(2,), dtype=np.float64, minimum=[-1.0, -0.5], maximum=[1.0, 0.5])

    clipped = action_to_native(DMControlAction(values=(2.0, -2.0)), action_spec)
    np.testing.assert_allclose(clipped, np.array([1.0, -0.5]))

    with pytest.raises(ValueError, match="shape"):
        action_to_native(DMControlAction(values=(0.0, 0.0, 0.0)), action_spec)


def test_dmcontrol_named_camera_config_applies_to_model():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import DMControlObservationProvider

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
        {"name": "fixed", "pose": (1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5), "fovy": 45.0},
    )

    np.testing.assert_allclose(physics.model.cam_pos[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(physics.model.cam_quat[0], np.array([0.5, 0.5, 0.5, 0.5]))
    assert physics.model.cam_fovy[0] == 45.0


def test_dmcontrol_free_camera_randomization_fails_loudly():
    from mllm_teleop_benchmark.sim.dmcontrol.observation import DMControlObservationProvider

    provider = DMControlObservationProvider(camera_manager=None)

    with pytest.raises(NotImplementedError, match="free-camera orbit"):
        provider._apply_mujoco_camera_config(
            object(),
            {"distance": 1.0, "azimuth": 10.0, "elevation": -30.0, "lookat": (0.0, 0.0, 0.0)},
        )


def test_dmcontrol_sim_adapter_worker_reset_step_shutdown():
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco")

    config = load_config("configs/dmcontrol_example.yaml")
    context = mp.get_context("spawn")
    action_queue: mp.Queue = context.Queue()
    result_queue: mp.Queue = context.Queue()
    worker = context.Process(target=sim_adapter_worker_entry, args=(config, action_queue, result_queue))
    worker.start()
    try:
        action_queue.put(PlayerCommand("reset"))
        initial = result_queue.get(timeout=15)
        assert isinstance(initial, PlayerRuntimeResult)
        if initial.error:
            pytest.skip(f"DMControl runtime unavailable: {initial.error}")
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
                }
            )
        )
        action_queue.put(PlayerCommand("step", sim_action))
        stepped = result_queue.get(timeout=15)
        assert isinstance(stepped, PlayerRuntimeResult)
        if stepped.error:
            pytest.skip(f"DMControl runtime step unavailable: {stepped.error}")
        assert stepped.observation is not None
        assert np.asarray(stepped.frame).shape[-1] == 3
        assert stepped.step_result is not None
        assert stepped.step_result.info["step_id"] == 1
    finally:
        action_queue.put(PlayerCommand("close"))
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
