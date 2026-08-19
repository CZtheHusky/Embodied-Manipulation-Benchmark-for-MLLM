import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml
from PIL import Image

from mllm_teleop_benchmark import BenchmarkRunner
from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.model_client import ModelResponse, ModelUsage
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.control.simpler_env import SimplerEnvDeltaEEFController
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_camera_manager,
    build_controller,
    build_mapping,
    build_observation_provider,
    build_sim,
    control_interface_metadata,
)
from mllm_teleop_benchmark.logging.episode_logger import make_experiment_group
from mllm_teleop_benchmark.mapping.frame_transform import RotationControlFrameTransform
from mllm_teleop_benchmark.core.types import SimplerEnvAction
from mllm_teleop_benchmark.sim.simpler_env.adapter import SimplerEnvSimAdapter


BASELINE_CONFIG = "configs/simpler_env_google_robot_pick_coke_can_mllm_chat_completions.yaml"
PERTURBED_CONFIG = "configs/simpler_env_google_robot_pick_coke_can_mllm_chat_completions_perturbed.yaml"
MLLM_CONFIGS = (
    (BASELINE_CONFIG, "fixed", "identity", True),
    (
        PERTURBED_CONFIG,
        "permuted_sign_flipped",
        "random_rotation_3d",
        False,
    ),
)
PUBLIC_CAMERAS = ("image/overhead_camera",)
PRIVATE_SENTINELS = {
    "state": "PRIVATE_STATE_SENTINEL",
    "reward": "PRIVATE_REWARD_SENTINEL",
    "success_oracle": "PRIVATE_SUCCESS_SENTINEL",
    "distance": "PRIVATE_DISTANCE_SENTINEL",
    "object_pose": "PRIVATE_OBJECT_POSE_SENTINEL",
    "eef_pose": "PRIVATE_EEF_POSE_SENTINEL",
    "native_sim_action": "PRIVATE_SIM_ACTION_SENTINEL",
    "seed": "PRIVATE_SEED_SENTINEL",
    "permutation": "PRIVATE_PERMUTATION_SENTINEL",
    "signs": "PRIVATE_SIGNS_SENTINEL",
    "rotation_sample": "PRIVATE_ROTATION_SENTINEL",
}


class FakeSimplerEnvClient:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    **{f"joystick_{index}": 0 for index in range(6)},
                    "joystick_6": 0.0,
                },
                "Reflection": (
                    "The current RGB views do not yet show a reliable motion "
                    "effect, so keep the pose unchanged and the gripper open."
                ),
            },
            separators=(",", ":"),
        )
        raw = {
            "id": f"fake-simpler-env-{len(self.requests)}",
            "model": "fake-simpler-env-model",
            "choices": [{"message": {"content": text}}],
        }
        return ModelResponse(
            text=text,
            request_id=raw["id"],
            model=raw["model"],
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=24,
                cached_input_tokens=20,
            ),
            raw=raw,
            latency_ms=1.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


class FakeInitiallySuccessfulSimplerEnv:
    def __init__(self) -> None:
        self.reset_observation = {
            "image": {
                "base_camera": {
                    "rgb": np.full((1, 4, 5, 3), 127, dtype=np.uint8),
                }
            }
        }
        self.get_info_calls = 0
        self.evaluate_calls = 0
        self.consecutive_grasp = 2
        self.lifted_obj = False
        self.episode_stats = {"grasped": False, "success_at": None}
        self.closed = False

    def reset(self, seed):
        return self.reset_observation, {}

    def get_wrapper_attr(self, name):
        if name != "get_info":
            raise AttributeError(name)
        return self.get_info

    def get_info(self):
        self.get_info_calls += 1
        return {
            "elapsed_steps": 0,
            **self.evaluate(),
        }

    def evaluate(self):
        self.evaluate_calls += 1
        self.consecutive_grasp += 1
        self.lifted_obj = True
        self.episode_stats["grasped"] = True
        self.episode_stats["success_at"] = 0
        return {
            "success": True,
            "episode_stats": {"grasped": True},
        }

    def close(self):
        self.closed = True


def test_simpler_env_provider_passes_the_audited_sensor_resolution():
    config = load_config(BASELINE_CONFIG)
    sim = build_sim(config)
    camera_manager = build_camera_manager(config.camera, config.benchmark.seed)
    provider = build_observation_provider(config, camera_manager)

    provider.configure_sim(sim)

    assert sim._camera_config == {
        "overhead_camera": {"width": 512, "height": 410}
    }


def test_simpler_env_native_make_merges_resolution_into_prepackaged_config(monkeypatch):
    module_names = (
        "mani_skill2_real2sim",
        "mani_skill2_real2sim.envs",
        "mani_skill2_real2sim.envs.custom_scenes",
    )
    for name in module_names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    leaf = ModuleType(
        "mani_skill2_real2sim.envs.custom_scenes.grasp_single_in_scene"
    )

    class FakeGraspSingleInSceneEnv:
        def _setup_prepackaged_env_init_config(self):
            return {
                "asset_root": "/pinned/assets",
                "camera_cfgs": {
                    "width": 640,
                    "height": 512,
                    "fov": 42,
                },
            }

    leaf.GraspSingleInSceneEnv = FakeGraspSingleInSceneEnv
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)

    calls = []

    class FakeSimplerEnv:
        @staticmethod
        def make(env_id, **kwargs):
            calls.append((env_id, kwargs))
            return object()

    adapter = SimplerEnvSimAdapter(load_config(BASELINE_CONFIG).simulator)
    adapter.set_camera_configuration(
        {"overhead_camera": {"width": 512, "height": 410}}
    )

    adapter._make_env(
        FakeSimplerEnv,
        {
            "max_episode_steps": 80,
            "camera_cfgs": {"add_segmentation": True},
        },
    )

    env_id, kwargs = calls[0]
    assert env_id == "google_robot_pick_coke_can"
    assert kwargs["asset_root"] == "/pinned/assets"
    assert kwargs["prepackaged_config"] is False
    assert kwargs["camera_cfgs"] == {
        "width": 512,
        "height": 410,
        "fov": 42,
        "add_segmentation": True,
    }


def _require_simpler_env() -> None:
    pytest.importorskip(
        "mani_skill2_real2sim",
        reason="pinned mani_skill2_real2sim runtime is not installed",
    )
    pytest.importorskip(
        "simpler_env",
        reason="pinned simpler_env runtime is not installed",
    )


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_simpler_env_mllm_configs_are_explicit_and_switchable(
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    config = load_config(config_path)
    schema = build_action_space(config).schema()
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 3
    assert config.logging.save_video is False
    assert config.task.params["reject_initial_success"] is True
    assert config.simulator.params["reset_evaluator_state_fields"] == [
        "consecutive_grasp",
        "lifted_obj",
        "episode_stats",
    ]
    assert config.simulator.params["absolute_gripper_target"] is True
    assert config.agent.observation.cameras == PUBLIC_CAMERAS
    assert tuple(camera.name for camera in config.camera.cameras) == PUBLIC_CAMERAS
    assert config.control_interface.active_dimensions == (
        "dx",
        "dy",
        "dz",
        "droll",
        "dpitch",
        "dyaw",
    )
    assert config.control_interface.gripper == "ternary"
    assert config.control_interface.mapping_strategy == mapping_strategy
    assert config.control_interface.frame_transform == frame_transform
    assert config.control_interface.mapping_visible_to_agent is visible
    assert config.control_interface.frame_visible_to_agent is visible
    assert tuple(schema) == tuple(f"joystick_{index}" for index in range(7))
    assert all(
        schema[f"joystick_{index}"]
        == {"kind": "continuous", "min": -30.0, "max": 30.0}
        for index in range(6)
    )
    assert schema["joystick_6"] == {
        "kind": "discrete",
        "values": [0.0, 0.5, 1.0],
    }

    control = raw["control_interface"]
    frame_description = control["control_frame_description"]
    assert "control origin at the TCP, not the tool frame" in frame_description
    assert "axis-angle rotation vectors both use robot-base axes" in frame_description
    assert "rotation is pre-multiplied about the TCP origin" in frame_description
    assert "fixed 180-degree yaw" in frame_description
    assert "base +x is world -x" in frame_description
    assert "base +y is world -y" in frame_description
    assert "base +z is world +z" in frame_description
    assert "normalize_action is false" in frame_description
    assert (
        "scaled to 0.01 meters and 0.0314159265358979 radians"
        in frame_description
    )
    assert "1.8 degrees" in frame_description
    assert "overhead_camera is mounted to link_camera" in frame_description
    assert "pose remains fixed for this arm-only benchmark" in frame_description
    assert "base +y projects approximately toward image left" in frame_description
    assert "base +x mainly away from the camera" in frame_description
    assert "upward image component" in frame_description
    assert "base +z toward image up" in frame_description
    assert "pointing partly toward the camera" in frame_description
    assert "base_camera" not in frame_description
    if visible:
        assert "with no sign compensation" in frame_description
    else:
        assert "no additional sign compensation is applied" in frame_description
    assert control["gripper_open_command"] == 0
    assert control["gripper_closed_command"] == 1
    assert control["gripper_hold_command"] == 0.5
    assert control["params"]["rotation_representation"] == "axis_angle"
    assert config.controller.frame == "ee_align"
    assert config.controller.scales == {
        "dx": 0.01,
        "dy": 0.01,
        "dz": 0.01,
        "droll": 0.0314159265358979,
        "dpitch": 0.0314159265358979,
        "dyaw": 0.0314159265358979,
    }

    group = make_experiment_group(config)
    assert group is not None
    assert "__output-" not in group
    if mapping_strategy == "fixed":
        assert "__mapping-" not in group
        assert "__frame-" not in group
    else:
        assert "mapping-permuted_sign_flipped" in group
        assert "frame-random_rotation_3d" in group


def test_simpler_env_mllm_task_text_is_identical_between_control_modes():
    baseline = load_config(BASELINE_CONFIG)
    perturbed = load_config(PERTURBED_CONFIG)

    assert baseline.task.params["instruction"] == perturbed.task.params["instruction"]
    assert baseline.task.params["instruction"] == "pick coke can."
    assert "joystick" not in baseline.task.params["instruction"].lower()
    assert "rotation" not in baseline.task.params["instruction"].lower()


def test_simpler_env_public_prompt_describes_controls_without_hidden_samples():
    baseline = load_config(BASELINE_CONFIG)
    baseline_prompt = ControlInstructionBuilder().build(
        control_interface_metadata(baseline, baseline.benchmark.seed),
        build_action_space(baseline).schema(),
    ).text
    perturbed = load_config(PERTURBED_CONFIG)
    perturbed_metadata = control_interface_metadata(
        perturbed,
        perturbed.benchmark.seed,
    )
    perturbed_prompt = ControlInstructionBuilder().build(
        perturbed_metadata,
        build_action_space(perturbed).schema(),
    ).text

    assert "dx, dy, and dz denote end-effector translation" in baseline_prompt
    assert "droll, dpitch, and dyaw denote end-effector rotation" in baseline_prompt
    assert "control origin at the TCP, not the tool frame" in baseline_prompt
    assert "axis-angle rotation vectors both use robot-base axes" in baseline_prompt
    assert "rotation is pre-multiplied about the TCP origin" in baseline_prompt
    assert "fixed 180-degree yaw" in baseline_prompt
    assert "base +x is world -x" in baseline_prompt
    assert "base +y is world -y" in baseline_prompt
    assert "base +z is world +z" in baseline_prompt
    assert "normalize_action is false" in baseline_prompt
    assert (
        "scaled to 0.01 meters and 0.0314159265358979 radians"
        in baseline_prompt
    )
    assert "1.8 degrees" in baseline_prompt
    assert "with no sign compensation" in baseline_prompt
    assert "overhead_camera is mounted to link_camera" in baseline_prompt
    assert "pose remains fixed for this arm-only benchmark" in baseline_prompt
    assert "base +y projects approximately toward image left" in baseline_prompt
    assert "base +x mainly away from the camera" in baseline_prompt
    assert "upward image component" in baseline_prompt
    assert "base +z toward image up" in baseline_prompt
    assert "pointing partly toward the camera" in baseline_prompt
    assert "base_camera" not in baseline_prompt
    assert "joystick_0 -> dx" in baseline_prompt
    assert "joystick_5 -> dyaw" in baseline_prompt
    assert "ternary control" in baseline_prompt
    assert "final virtual joystick" in baseline_prompt
    assert "1 selects the absolute closed target" in baseline_prompt
    assert "0.5 preserves the target selected" in baseline_prompt
    assert "No additional control-frame rotation is applied" in baseline_prompt

    assert "exact joystick-to-dimension mapping is not disclosed" in perturbed_prompt
    assert "exact physical-frame transform is not disclosed" in perturbed_prompt
    assert "control origin at the TCP, not the tool frame" in perturbed_prompt
    assert "axis-angle rotation vectors both use robot-base axes" in perturbed_prompt
    assert "rotation is pre-multiplied about the TCP origin" in perturbed_prompt
    assert (
        "scaled to 0.01 meters and 0.0314159265358979 radians"
        in perturbed_prompt
    )
    assert "1.8 degrees" in perturbed_prompt
    assert "no additional sign compensation is applied" in perturbed_prompt
    assert "fixed 180-degree yaw" in perturbed_prompt
    assert "overhead_camera is mounted to link_camera" in perturbed_prompt
    assert "pose remains fixed for this arm-only benchmark" in perturbed_prompt
    assert "base +y projects approximately toward image left" in perturbed_prompt
    assert "base +x mainly away from the camera" in perturbed_prompt
    assert "base +z toward image up" in perturbed_prompt
    assert "1 selects the absolute closed target" in perturbed_prompt
    assert "0.5 preserves the target selected" in perturbed_prompt
    assert "base_camera" not in perturbed_prompt
    assert (
        "Infer the hidden controls from visual observations and your own action "
        "history"
    ) in perturbed_prompt
    assert "joystick_0 ->" not in perturbed_prompt
    assert "mapping" not in perturbed_metadata
    assert "mapping_strategy" not in perturbed_metadata
    assert "frame_transform" not in perturbed_metadata


def test_simpler_env_perturbation_keeps_gripper_unmodified():
    config = load_config(PERTURBED_CONFIG)
    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)

    assert isinstance(controller, SimplerEnvDeltaEEFController)
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

    action = JoystickAction(
        {
            **{f"joystick_{index}": 0.0 for index in range(6)},
            "joystick_6": 1.0,
        }
    )
    sim_action = controller.translate(action)
    assert np.asarray(sim_action.values).shape == (7,)
    assert sim_action.values[-1] == 1.0
    assert controller.metadata()["mapping"] == {
        "type": "fixed",
        "visible_to_agent": False,
    }
    assert controller.metadata()["frame_transform"] == {
        "type": "random_rotation_3d",
        "visible_to_agent": False,
    }


def test_simpler_env_adapter_serializes_pose_like_info():
    config = load_config(BASELINE_CONFIG)
    adapter = SimplerEnvSimAdapter(config.simulator)
    pose = type(
        "PoseLike",
        (),
        {
            "p": np.asarray([0.1, 0.2, 0.3]),
            "q": np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
    )()

    assert adapter._jsonable({"tcp_pose": pose}) == {
        "tcp_pose": {
            "position": [0.1, 0.2, 0.3],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    }


def test_simpler_env_absolute_gripper_target_uses_one_native_step():
    class FakeActionSpace:
        shape = (7,)
        low = np.full(7, -1.0, dtype=np.float32)
        high = np.full(7, 1.0, dtype=np.float32)

    class FakeEnv:
        def __init__(self):
            self.action_space = FakeActionSpace()
            self.actions = []
            self.unwrapped = self
            self.gripper = type(
                "FakeGripper",
                (),
                {
                    "config": type(
                        "FakeGripperConfig",
                        (),
                        {
                            "use_target": True,
                            "clip_target": True,
                            "small_action_repeat_last_target": True,
                            "clip_target_thres": 0.01,
                        },
                    )(),
                    "articulation": type(
                        "FakeArticulation",
                        (),
                        {
                            "get_qlimits": lambda _self: np.asarray(
                                [[0.0, 1.3], [0.0, 1.3]]
                            )
                        },
                    )(),
                    "joint_indices": np.asarray([0, 1]),
                    "target": np.asarray([0.0, 0.0]),
                    "set_state": lambda self, state: setattr(
                        self,
                        "target",
                        np.asarray(state["target_qpos"]).copy(),
                    ),
                    "get_state": lambda self: {
                        "target_qpos": np.asarray(self.target).copy()
                    },
                },
            )()
            self.agent = type(
                "FakeAgent",
                (),
                {
                    "controller": type(
                        "FakeCombinedController",
                        (),
                        {"controllers": {"gripper": self.gripper}},
                    )()
                },
            )()

        def step(self, action):
            values = np.asarray(action, dtype=np.float32).copy()
            self.actions.append(values)
            return {}, 0.0, False, False, {}

    config = load_config(BASELINE_CONFIG)
    adapter = SimplerEnvSimAdapter(config.simulator)
    fake_env = FakeEnv()
    adapter.env = fake_env

    open_action = SimplerEnvAction(
        values=(0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    )
    opened = adapter.step(open_action)
    np.testing.assert_allclose(fake_env.gripper.target, (-0.01, -0.01))
    np.testing.assert_allclose(fake_env.actions[0][:-1], open_action.values[:-1])
    assert fake_env.actions[0][-1] == 0.0
    assert opened.info["step_id"] == 1
    assert opened.info["native_step_id"] == 1
    assert opened.info["native_steps_this_action"] == 1
    np.testing.assert_allclose(
        opened.info["benchmark_applied_native_action"]["values"],
        fake_env.actions[0],
    )
    assert opened.info["benchmark_applied_native_action"]["shape"] == [7]
    assert opened.info["benchmark_absolute_gripper"] == {
        "controller_command": -1.0,
        "applied_env_command": 0.0,
        "target_update": "open",
        "target_qpos": [-0.01, -0.01],
    }

    close_action = SimplerEnvAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    closed = adapter.step(close_action)
    np.testing.assert_allclose(fake_env.gripper.target, (1.31, 1.31))
    assert fake_env.actions[1][-1] == 0.0
    assert closed.info["benchmark_absolute_gripper"] == {
        "controller_command": 1.0,
        "applied_env_command": 0.0,
        "target_update": "closed",
        "target_qpos": [1.31, 1.31],
    }

    hold_action = SimplerEnvAction(values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    held = adapter.step(hold_action)
    np.testing.assert_allclose(fake_env.gripper.target, (1.31, 1.31))
    assert held.info["step_id"] == 3
    assert held.info["native_step_id"] == 3
    assert held.info["benchmark_absolute_gripper"] == {
        "controller_command": 0.0,
        "applied_env_command": 0.0,
        "target_update": "preserved",
        "target_qpos": [1.31, 1.31],
    }

    fake_env.gripper.config.use_target = False
    with pytest.raises(RuntimeError, match="pinned target-delta controller"):
        adapter.step(close_action)
    assert len(fake_env.actions) == 3


def test_simpler_env_rejects_evaluator_success_before_model_request(
    monkeypatch,
    tmp_path,
):
    client = FakeSimplerEnvClient()
    fake_env = FakeInitiallySuccessfulSimplerEnv()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-simpler-env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    def install_fake_env(adapter):
        if adapter.env is None:
            adapter.env = fake_env

    monkeypatch.setattr(SimplerEnvSimAdapter, "_ensure_env", install_fake_env)
    config = load_config(BASELINE_CONFIG)
    config = replace(
        config,
        logging=replace(
            config.logging,
            output_dir=str(tmp_path / "initial-success"),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="already successful immediately after reset",
    ):
        BenchmarkRunner(config).run((0,))

    assert "success" not in fake_env.reset_observation
    assert fake_env.get_info_calls == 1
    assert fake_env.evaluate_calls == 1
    assert fake_env.consecutive_grasp == 2
    assert fake_env.lifted_obj is False
    assert fake_env.episode_stats == {"grasped": False, "success_at": None}
    assert fake_env.closed is True
    assert client.requests == []


def test_simpler_env_runtime_rgb_and_native_action_contract():
    _require_simpler_env()
    config = load_config(BASELINE_CONFIG)
    sim = build_sim(config)
    provider = build_observation_provider(
        config,
        build_camera_manager(config.camera, config.benchmark.seed),
    )
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    provider.configure_sim(sim)

    try:
        sim.reset(config.task, seed=config.benchmark.seed)
        unwrapped = getattr(sim.env, "unwrapped", sim.env)
        base_quaternion_wxyz = np.asarray(unwrapped.agent.robot.pose.q)
        np.testing.assert_allclose(
            np.abs(base_quaternion_wxyz),
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-7,
        )
        observation = provider.get_observation(sim)
        assert set(PUBLIC_CAMERAS).issubset(observation.images)
        for camera in PUBLIC_CAMERAS:
            image = np.asarray(observation.images[camera])
            assert image.shape == (410, 512, 3)
            assert image.dtype == np.uint8
            assert float(image.mean()) > 1.0
            assert float(image.std()) > 1.0

        action = JoystickAction(
            {
                **{f"joystick_{index}": 0.0 for index in range(6)},
                "joystick_6": 0.0,
            }
        )
        sim_action = controller.translate(action)
        native_action, clip_metadata = sim._native_action(sim_action)
        assert native_action.shape == (7,)
        np.testing.assert_array_less(
            np.asarray(sim.env.action_space.low) - 1e-7,
            native_action,
        )
        np.testing.assert_array_less(
            native_action,
            np.asarray(sim.env.action_space.high) + 1e-7,
        )
        assert clip_metadata["clipped"] is False

        step = sim.step(sim_action)
        assert step.observation.step_id == 1
        next_observation = provider.get_observation(sim)
        assert set(PUBLIC_CAMERAS).issubset(next_observation.images)
        assert all(
            float(np.asarray(next_observation.images[camera]).std()) > 1.0
            for camera in PUBLIC_CAMERAS
        )
    finally:
        sim.close()


def test_simpler_env_real_gripper_command_sets_absolute_target():
    _require_simpler_env()
    config = load_config(BASELINE_CONFIG)
    sim = build_sim(config)
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )

    def translate_gripper(value):
        return controller.translate(
            JoystickAction(
                {
                    **{f"joystick_{index}": 0.0 for index in range(6)},
                    "joystick_6": value,
                }
            )
        )

    try:
        sim.reset(config.task, seed=config.benchmark.seed)
        native_gripper = sim.env.unwrapped.agent.controller.controllers["gripper"]
        joint_limits = native_gripper.articulation.get_qlimits()[
            native_gripper.joint_indices
        ]
        clip_threshold = native_gripper.config.clip_target_thres

        sim.step(translate_gripper(0.0))
        open_target = np.asarray(native_gripper._target_qpos).copy()
        sim.step(translate_gripper(1.0))
        closed_target = np.asarray(native_gripper._target_qpos).copy()
        sim.step(translate_gripper(0.5))
        held_target = np.asarray(native_gripper._target_qpos).copy()

        np.testing.assert_allclose(
            open_target,
            joint_limits[:, 0] - clip_threshold,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            closed_target,
            joint_limits[:, 1] + clip_threshold,
            atol=1e-7,
        )
        np.testing.assert_allclose(held_target, closed_target, atol=1e-7)
        assert sim.get_state()["native_step_id"] == 3
    finally:
        sim.close()


def test_simpler_env_native_time_limit_does_not_preempt_agent_command_budget():
    _require_simpler_env()
    config = load_config(BASELINE_CONFIG)
    sim = build_sim(config)
    controller = build_controller(
        config,
        build_mapping(config, config.benchmark.seed),
        config.benchmark.seed,
    )
    hold = controller.translate(
        JoystickAction(
            {
                **{f"joystick_{index}": 0.0 for index in range(6)},
                "joystick_6": 0.5,
            }
        )
    )

    try:
        sim.reset(config.task, seed=config.benchmark.seed)
        result = None
        for _ in range(config.task.max_steps):
            result = sim.step(hold)
            if result.terminated or result.truncated:
                break

        assert result is not None
        assert result.terminated is False
        assert result.truncated is False
        assert result.info["step_id"] == 80
        assert result.info["native_step_id"] == 80
        assert result.info["elapsed_steps"] == 80
    finally:
        sim.close()


def test_simpler_env_pick_coke_can_experiment_seeds_reset_canary():
    _require_simpler_env()
    config = load_config(BASELINE_CONFIG)
    sim = build_sim(config)
    provider = build_observation_provider(
        config,
        build_camera_manager(config.camera, config.benchmark.seed),
    )

    try:
        for seed in range(config.benchmark.seed, config.benchmark.seed + 10):
            sim.reset(config.task, seed=seed)
            state = sim.get_state()
            assert state["success"] is False, (
                f"PickCokeCan seed {seed} is successful immediately after reset"
            )

            observation = provider.get_observation(sim)
            assert set(PUBLIC_CAMERAS).issubset(observation.images)
            for camera in PUBLIC_CAMERAS:
                image = np.asarray(observation.images[camera])
                assert image.ndim == 3
                assert image.shape[-1] == 3
                assert image.dtype == np.uint8
                assert np.any(image != 0), (
                    f"PickCokeCan seed {seed} camera {camera} is all black"
                )
    finally:
        sim.close()


@pytest.mark.parametrize(
    ("config_path", "mapping_strategy", "frame_transform", "visible"),
    MLLM_CONFIGS,
)
def test_simpler_env_fake_client_closes_public_rgb_loop_without_private_state(
    monkeypatch,
    tmp_path,
    config_path,
    mapping_strategy,
    frame_transform,
    visible,
):
    _require_simpler_env()
    client = FakeSimplerEnvClient()
    monkeypatch.setenv("MLLM_TELEOP_MODEL", "fake-simpler-env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_API_KEY_SENTINEL")
    monkeypatch.setattr(
        "mllm_teleop_benchmark.core.registry.OpenAICompatibleClient",
        lambda **_kwargs: client,
    )

    original_get_state = SimplerEnvSimAdapter.get_state

    def poisoned_get_state(self):
        state = original_get_state(self)
        state.update(PRIVATE_SENTINELS)
        return state

    monkeypatch.setattr(SimplerEnvSimAdapter, "get_state", poisoned_get_state)
    config = load_config(config_path)
    config = replace(
        config,
        task=replace(config.task, max_steps=4, success_threshold=2.0),
        logging=replace(
            config.logging,
            output_dir=str(tmp_path / mapping_strategy),
            save_video=False,
        ),
    )

    episode = BenchmarkRunner(config).run((0,)).episodes[0]

    assert episode.steps == 4
    assert episode.termination_reason == "max_steps"
    assert len(client.requests) == 4
    assert [_request_image_count(request) for request in client.requests] == [
        1,
        2,
        3,
        4,
    ]
    assert [
        sum(message["role"] == "assistant" for message in request.messages)
        for request in client.requests
    ] == [0, 1, 2, 3]

    request_text = "\n".join(
        _request_text(request)
        for request in client.requests
    )
    assert "Camera order: image/overhead_camera" in request_text
    assert "image/base_camera" not in request_text
    assert "Segmentation" not in request_text
    assert "pick coke can" in request_text
    assert "success" not in request_text.lower()
    assert "benchmark_applied_native_action" not in request_text
    assert "benchmark_absolute_gripper" not in request_text
    for forbidden in (
        *PRIVATE_SENTINELS.values(),
        "PRIVATE_API_KEY_SENTINEL",
        "permuted_sign_flipped",
        "random_rotation_3d",
        "rotation_matrix",
        "quaternion_xyzw",
    ):
        assert forbidden not in request_text

    episode_dir = Path(episode.log_dir)
    trajectory = [
        json.loads(line)
        for line in (episode_dir / "trajectory.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(trajectory) == 4
    assert all(step["validation"]["is_valid"] for step in trajectory)
    assert all(step["reflection"] for step in trajectory)
    assert all(len(step["sim_action"]["values"]) == 7 for step in trajectory)
    assert all(
        np.all(np.isfinite(step["sim_action"]["values"]))
        for step in trajectory
    )
    assert all(
        len(step["info"]["benchmark_applied_native_action"]["values"]) == 7
        for step in trajectory
    )
    assert all(
        step["info"]["benchmark_applied_native_action"]["values"][-1] == 0.0
        for step in trajectory
    )
    assert all(
        "target_qpos" in step["info"]["benchmark_absolute_gripper"]
        for step in trajectory
    )

    pngs = sorted(episode_dir.glob("model_calls/images/step_*/*.png"))
    assert len(pngs) == 4
    assert len({path.read_bytes() for path in pngs}) == 4
    for path in pngs:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        assert pixels.ndim == 3
        assert float(pixels.mean()) > 1.0
        assert float(pixels.std()) > 1.0

    manifest = json.loads(
        (episode_dir.parent / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model"]["api"] == "chat_completions"
    assert manifest["model"]["reasoning_effort"] == "low"
    assert "action_output_mode" not in manifest["model"]
    assert manifest["model"]["prompt_version"] == "mllm_joystick_chat_completions_v10"
    assert manifest["simulator"]["type"] == "simpler_env"
    assert manifest["history"]["action_window"] == "all"
    assert manifest["history"]["image_window"] == 3
    assert manifest["control_interface"]["mapping_strategy"] == mapping_strategy
    assert manifest["control_interface"]["frame_transform"] == frame_transform
    assert manifest["control_interface"]["mapping_visible_to_agent"] is visible
    assert manifest["control_interface"]["frame_visible_to_agent"] is visible

    for artifact in episode_dir.rglob("*"):
        if artifact.is_file():
            contents = artifact.read_bytes()
            assert b"PRIVATE_API_KEY_SENTINEL" not in contents
    for request_path in episode_dir.glob("model_calls/step_*/request.json"):
        contents = request_path.read_bytes()
        for sentinel in PRIVATE_SENTINELS.values():
            assert sentinel.encode() not in contents


def _request_image_count(request) -> int:
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
