from dataclasses import asdict
import json

import numpy as np
import pytest

from mllm_teleop_benchmark.action.joystick import JoystickAction
from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.model_client import (
    ModelClient,
    ModelResponse,
    ModelUsage,
)
from mllm_teleop_benchmark.agents.model_policy import MLLMPolicy
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.registry import (
    build_action_space,
    build_controller,
    build_mapping,
    control_interface_metadata,
)
from mllm_teleop_benchmark.core.types import Observation, RoboCasaAction
from mllm_teleop_benchmark.tasks.robocasa_task import RoboCasaGenericTask


HUMAN_CONFIG = "configs/robocasa365_cutting_tool_selection.yaml"
BASELINE_CONFIG = (
    "configs/robocasa365_cutting_tool_selection_mllm_reflection.yaml"
)
PERTURBED_CONFIG = (
    "configs/robocasa365_cutting_tool_selection_mllm_reflection_perturbed.yaml"
)
MLLM_CONFIGS = (BASELINE_CONFIG, PERTURBED_CONFIG)
PUBLIC_CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
ASSET_REVISIONS = {
    "robocasa_assets": "1b92c3d02ca4354984fec961357db0bff7b32166",
    "lightwheel": "420a04af939c34873e6839a586b70844baf28aab",
}
ACTIVE_DIMENSIONS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
CONTROLLER_SCALES = {
    "dx": 1.0,
    "dy": 1.0,
    "dz": 1.0,
    "droll": 1.0,
    "dpitch": 1.0,
    "dyaw": 1.0,
    "gripper": 1.0,
}


class RecordingClient(ModelClient):
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        text = json.dumps(
            {
                "channels": {
                    f"joystick_{index}": 0
                    for index in range(7)
                },
                "Reflection": "The controls remain under visual evaluation.",
            }
        )
        return ModelResponse(
            text=text,
            request_id="static-robocasa365-request",
            model="static-robocasa365-model",
            usage=ModelUsage(input_tokens=1, output_tokens=1),
            raw={"choices": [{"message": {"content": text}}]},
            latency_ms=0.0,
            attempts=1,
            request_bytes=len(request.encoded_body()),
        )


@pytest.mark.parametrize(
    ("config_path", "agent_type"),
    (
        (HUMAN_CONFIG, "random"),
        (BASELINE_CONFIG, "mllm"),
        (PERTURBED_CONFIG, "mllm"),
    ),
)
def test_robocasa365_configs_load_with_exact_task_metadata(
    config_path,
    agent_type,
):
    config = load_config(config_path)

    assert config.benchmark.num_episodes == 10
    assert config.simulator.type == "robocasa"
    assert config.simulator.env_id == "robocasa/CuttingToolSelection"
    assert config.simulator.env_kwargs == {
        "split": "target",
        "obj_registries": ["objaverse", "lightwheel"],
        "camera_widths": 256,
        "camera_heights": 256,
        "randomize_cameras": False,
    }
    assert config.simulator.params == {"asset_revisions": ASSET_REVISIONS}
    assert config.task.type == "robocasa_generic"
    assert config.task.max_steps == 800
    assert config.task.success_threshold == 1.0
    assert config.task.params["instruction"] == (
        "Place the appropriate cutting tool for the food on the cutting board."
    )
    assert config.task.params["instruction_from_observation"] is True
    assert (
        config.task.params["instruction_observation_key"]
        == "annotation.human.task_description"
    )
    assert config.task.params.get("reject_initial_success", False) is (
        agent_type == "mllm"
    )
    assert tuple(camera.name for camera in config.camera.cameras) == PUBLIC_CAMERAS
    assert config.agent.type == agent_type
    assert config.logging.save_video is False
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.frame_camera == PUBLIC_CAMERAS[0]

    interface = config.control_interface
    assert interface.active_dimensions == ACTIVE_DIMENSIONS
    assert interface.gripper == "binary"
    assert interface.channel_kind == "discrete"
    assert interface.channel_values == (-1.0, 0.0, 1.0)
    assert interface.gripper_open_command == 0.0
    assert interface.gripper_hold_command is None
    assert interface.gripper_closed_command == 1.0
    assert interface.params == {
        "scale_mode": "robocasa_normalized_pandaomron_osc_base_delta",
        "control_site": "robot0_right_center",
        "control_frame": "robot_base",
        "frame_moves_with_base": True,
        "native_translation_gain_m": 0.05,
        "native_rotation_gain_rad": 0.5,
        "sign_compensation": False,
    }
    assert config.controller.type == "robocasa_osc"
    assert config.controller.scales == CONTROLLER_SCALES
    assert config.controller.params == {"control_mode": 1.0}

    if agent_type == "mllm":
        assert config.agent.action_output_mode == "action_with_reflection"
        assert config.agent.model is not None
        assert config.agent.model.api == "chat_completions"
        assert config.agent.model.reasoning_effort == "low"
        assert config.agent.observation.cameras == PUBLIC_CAMERAS
        assert config.agent.history.action_window == "all"
        assert config.agent.history.image_window == 3


def test_robocasa365_action_channels_and_native_base_contract_are_exact():
    config = load_config(BASELINE_CONFIG)
    action_schema = build_action_space(config).schema()

    assert tuple(action_schema) == tuple(
        f"joystick_{index}" for index in range(7)
    )
    assert all(
        action_schema[f"joystick_{index}"]
        == {"kind": "discrete", "values": [-1.0, 0.0, 1.0]}
        for index in range(6)
    )
    assert action_schema["joystick_6"] == {
        "kind": "discrete",
        "values": [0.0, 1.0],
    }

    mapping = build_mapping(config, config.benchmark.seed)
    controller = build_controller(config, mapping, config.benchmark.seed)
    action = JoystickAction(
        {
            "joystick_0": 1.0,
            "joystick_1": -1.0,
            "joystick_2": 0.0,
            "joystick_3": 1.0,
            "joystick_4": 0.0,
            "joystick_5": -1.0,
            "joystick_6": 1.0,
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
    np.testing.assert_allclose(
        sim_action.values["action.end_effector_position"],
        [1.0, -1.0, 0.0],
    )
    np.testing.assert_allclose(
        sim_action.values["action.end_effector_rotation"],
        [1.0, 0.0, -1.0],
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

    perturbed = load_config(PERTURBED_CONFIG)
    hidden_mapping = build_mapping(perturbed, perturbed.benchmark.seed)
    assert set(hidden_mapping.channels.values()) == {
        *ACTIVE_DIMENSIONS,
        "gripper",
    }
    assert hidden_mapping.channels["joystick_6"] == "gripper"
    assert hidden_mapping.signs["joystick_6"] == 1.0


def test_robocasa365_task_uses_only_dynamic_public_annotation():
    config = load_config(BASELINE_CONFIG)
    task = RoboCasaGenericTask(config.task)
    public_instruction = (
        "Place the appropriate cutting tool for cutting the cucumber skin on "
        "the cutting board."
    )

    class StubSim:
        def reset(self, task_config, seed):
            assert task_config is config.task
            assert seed == config.benchmark.seed
            return Observation(
                step_id=0,
                images={},
                state={
                    "annotation.human.task_description": public_instruction,
                    "state.object_pose": "PRIVATE_OBJECT_POSE",
                    "reward": "PRIVATE_REWARD",
                    "is_success": "PRIVATE_SUCCESS_ORACLE",
                },
            )

        def get_state(self):
            return {"is_success": False}

    task.reset(StubSim(), config.benchmark.seed)

    assert task.instruction() == public_instruction
    assert "PRIVATE_" not in task.instruction()


def test_robocasa365_perturbed_prompt_excludes_private_and_audit_metadata():
    config = load_config(PERTURBED_CONFIG)
    action_space = build_action_space(config)
    public_instruction = (
        "Place the appropriate cutting tool for cutting the cucumber skin on "
        "the cutting board."
    )
    observation = Observation(
        step_id=0,
        images={
            camera: np.zeros((8, 8, 3), dtype=np.uint8)
            for camera in PUBLIC_CAMERAS
        },
        state={
            "state.object_pose": "PRIVATE_OBJECT_POSE",
            "state.end_effector_pose": "PRIVATE_EEF_POSE",
            "reward": "PRIVATE_REWARD",
            "is_success": "PRIVATE_SUCCESS_ORACLE",
            "native_sim_action": "PRIVATE_NATIVE_ACTION",
        },
        metadata={"episode_seed": "PRIVATE_EPISODE_SEED"},
    )
    context = AgentContext(
        observation=observation,
        task_instruction=public_instruction,
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={
            "control_interface": control_interface_metadata(
                config,
                config.benchmark.seed,
            )
        },
    )
    client = RecordingClient()
    policy = MLLMPolicy(config.agent, client, "static-robocasa365-model")

    policy.act(context)

    request_text = client.requests[0].encoded_body().decode("utf-8")
    assert public_instruction in request_text
    assert "exact joystick-to-dimension mapping is not disclosed" in request_text
    assert "exact physical-frame transform is not disclosed" in request_text
    assert "joystick_0 ->" not in request_text
    for forbidden in (
        "PRIVATE_",
        "state.object_pose",
        "state.end_effector_pose",
        "native_sim_action",
        "permuted_sign_flipped",
        "random_rotation_3d",
        "obj_registries",
        "objaverse",
        "lightwheel",
        *ASSET_REVISIONS.values(),
    ):
        assert forbidden not in request_text


def test_robocasa365_mllm_conditions_differ_only_in_control_complexity():
    baseline = asdict(load_config(BASELINE_CONFIG))
    perturbed = asdict(load_config(PERTURBED_CONFIG))

    assert baseline["control_interface"]["mapping_strategy"] == "fixed"
    assert baseline["control_interface"]["mapping_visible_to_agent"] is True
    assert baseline["control_interface"]["frame_transform"] == "identity"
    assert baseline["control_interface"]["frame_visible_to_agent"] is True
    assert (
        perturbed["control_interface"]["mapping_strategy"]
        == "permuted_sign_flipped"
    )
    assert perturbed["control_interface"]["mapping_visible_to_agent"] is False
    assert (
        perturbed["control_interface"]["frame_transform"]
        == "random_rotation_3d"
    )
    assert perturbed["control_interface"]["frame_visible_to_agent"] is False

    for data in (baseline, perturbed):
        data.pop("raw")
        data["source_path"] = "<condition>"
        data["benchmark"]["name"] = "<condition>"
        data["logging"]["output_dir"] = "<condition>"
        data["mapping"] = "<derived-condition>"
        data["frame_transform"] = "<derived-condition>"
        data["control_interface"]["mapping_strategy"] = "<condition>"
        data["control_interface"]["mapping_visible_to_agent"] = "<condition>"
        data["control_interface"]["frame_transform"] = "<condition>"
        data["control_interface"]["frame_visible_to_agent"] = "<condition>"

    assert baseline == perturbed
