import pytest

from mllm_teleop_benchmark.action.parser import ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.prompt import PromptBuilder
from mllm_teleop_benchmark.config.schema import ControlInterfaceConfig
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.core.episode import EpisodeManager
from mllm_teleop_benchmark.core.registry import build_action_space, build_mapping, control_interface_metadata
from mllm_teleop_benchmark.core.types import Observation


def test_reacher_control_interface_exposes_two_joysticks_without_gripper():
    config = load_config("configs/reacher_delta_eef.yaml")
    action_space = build_action_space(config)
    assert tuple(action_space.channels) == ("joystick_0", "joystick_1")
    assert build_mapping(config).metadata()["channels"] == {"joystick_0": "dx", "joystick_1": "dy"}

    valid = ActionParser().parse({"joystick_0": 0, "joystick_1": 1})
    assert ActionValidator(action_space).validate(valid).is_valid
    extra = ActionParser().parse({"joystick_0": 0, "joystick_1": 1, "joystick_6": 0})
    assert not ActionValidator(action_space).validate(extra).is_valid


def test_control_interface_prompt_mentions_visible_two_axis_mapping():
    config = load_config("configs/reacher_delta_eef.yaml")
    action_space = build_action_space(config)
    context = AgentContext(
        observation=Observation(step_id=0, images={}, state={}, metadata={}),
        task_instruction="Move to the target.",
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={"control_interface": control_interface_metadata(config)},
    )
    payload = PromptBuilder().build(context)
    assert "2 virtual joystick" in payload.control_instruction
    assert "active controllable dimensions are: dx, dy" in payload.control_instruction
    assert "joystick_0 -> dx" in payload.control_instruction
    assert "joystick_1 -> dy" in payload.control_instruction
    assert "No additional control-frame rotation is applied" in payload.control_instruction


def test_hidden_control_interface_prompt_does_not_expose_mapping():
    interface = ControlInterfaceConfig(
        mode="eef_delta_pose",
        active_dimensions=("dx", "dy"),
        visible_metadata=False,
        mapping_visible_to_agent=False,
    )
    action_space = ActionSpace.from_config(interface.to_action_space_config())
    context = AgentContext(
        observation=Observation(step_id=0, images={}, state={}, metadata={}),
        task_instruction="Move to the target.",
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={"control_interface": {**interface.metadata(), "source": "control_interface"}},
    )
    payload = PromptBuilder().build(context)
    assert "joystick-to-dimension mapping is not visible" in payload.control_instruction
    assert "dx" not in payload.control_instruction
    assert "dy" not in payload.control_instruction


def test_motion_confusion_keeps_gripper_as_final_joystick():
    interface = ControlInterfaceConfig(
        mode="eef_delta_pose",
        active_dimensions=("dx", "dy", "dz"),
        gripper="direct",
        mapping_strategy="permuted_sign_flipped",
    )
    mapping = interface.to_mapping_config(seed=123)
    assert tuple(mapping.channels)[-1] == "joystick_3"
    assert mapping.channels["joystick_3"] == "gripper"
    assert mapping.signs["joystick_3"] == 1.0
    motion_dims = {dim for channel, dim in mapping.channels.items() if channel != "joystick_3"}
    assert motion_dims == {"dx", "dy", "dz"}


def test_combined_mapping_confusion_always_applies_permutation_and_sign_flip():
    interface = ControlInterfaceConfig(
        mode="eef_delta_pose",
        active_dimensions=("dx", "dy"),
        mapping_strategy="permuted_sign_flipped",
    )

    for seed in range(20):
        mapping = interface.to_mapping_config(seed=seed)
        assert list(mapping.channels.values()) == ["dy", "dx"]
        assert any(sign < 0.0 for sign in mapping.signs.values())


def test_control_instruction_panel_lines_include_mode_mapping_frame_and_scale():
    config = load_config("configs/reacher_delta_eef.yaml")
    action_space = build_action_space(config)
    instruction = ControlInstructionBuilder().build(control_interface_metadata(config), action_space.schema())
    text = "\n".join(instruction.panel_lines)
    assert "control mode: eef_delta_pose" in text
    assert "mapping complexity: none" in text
    assert "frame complexity: none" in text
    assert "scales:" in text


def test_control_interface_frame_transform_override_drives_metadata():
    config = load_config(
        "configs/reacher_delta_eef.yaml",
        ("control_interface.frame_transform=rotation_2d", "control_interface.params.frame_transform.angle_degrees=90"),
    )
    metadata = control_interface_metadata(config)
    assert metadata["frame_complexity_present"] is True
    assert "frame_transform" not in metadata
    assert "frame_transform_metadata" not in metadata


def test_control_instruction_does_not_reveal_complex_mapping_or_signs():
    config = load_config(
        "configs/pusher_delta_eef.yaml",
        ("control_interface.mapping_strategy=permuted_sign_flipped", "control_interface.frame_transform=random_rotation_3d"),
    )
    action_space = build_action_space(config)
    metadata = control_interface_metadata(config, seed=3)
    instruction = ControlInstructionBuilder().build(metadata, action_space.schema())
    text = instruction.text + "\n" + "\n".join(instruction.panel_lines)
    assert "mapping complexity: present" in text
    assert "frame complexity: present" in text
    assert "dx, dy, and dz denote end-effector translation" in text
    assert "positive mapped value commands the positive direction" in text
    assert "No additional control-frame rotation is applied" not in text
    assert (
        "hidden joystick mapping and control-frame transform remain fixed "
        "for the entire episode"
    ) in text
    assert (
        "Infer the hidden controls from visual observations and your own action "
        "history"
    ) in text
    assert "no worked example or sampled hidden-control value is provided" in text
    assert "permuted_sign_flipped" not in text
    assert "random_rotation_3d" not in text
    assert "joystick_0 ->" not in text
    assert "joystick_1 ->" not in text
    assert "mapping_signs" not in metadata
    assert "mapping" not in metadata
    assert "mapping_strategy" not in metadata
    assert "frame_transform" not in metadata
    assert "rotation_matrix" not in str(metadata)


def test_control_interface_strips_public_frame_description():
    interface = ControlInterfaceConfig(
        mode="eef_delta_pose",
        active_dimensions=("dx", "dy", "dz"),
        control_frame_description="  the public world frame  ",
    )

    assert interface.control_frame_description == "the public world frame"
    assert interface.metadata()["control_frame_description"] == (
        "the public world frame"
    )


def test_control_interface_rejects_blank_public_frame_description():
    with pytest.raises(ValueError, match="control_frame_description must not be blank"):
        ControlInterfaceConfig(
            mode="eef_delta_pose",
            active_dimensions=("dx",),
            control_frame_description=" \t ",
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("gripper_open_command", -0.01),
        ("gripper_closed_command", 1.01),
        ("gripper_hold_command", float("nan")),
    ),
)
def test_direct_gripper_prompt_commands_must_fit_virtual_schema(field, value):
    kwargs = {
        "mode": "eef_delta_pose",
        "active_dimensions": ("dx",),
        "gripper": "direct",
        field: value,
    }

    with pytest.raises(ValueError, match=r"virtual gripper schema \[0, 1\]"):
        ControlInterfaceConfig(**kwargs)


def test_binary_gripper_prompt_commands_must_be_schema_values():
    with pytest.raises(ValueError, match="must be one of"):
        ControlInterfaceConfig(
            mode="eef_delta_pose",
            active_dimensions=("dx",),
            gripper="binary",
            gripper_hold_command=0.5,
        )


def test_gripper_open_and_closed_prompt_commands_must_differ():
    with pytest.raises(ValueError, match="open and closed commands must differ"):
        ControlInterfaceConfig(
            mode="eef_delta_pose",
            active_dimensions=("dx",),
            gripper="direct",
            gripper_open_command=0.5,
            gripper_closed_command=0.5,
        )


def test_gripper_prompt_commands_require_virtual_gripper_channel():
    with pytest.raises(ValueError, match="require a virtual gripper channel"):
        ControlInterfaceConfig(
            mode="eef_delta_pose",
            active_dimensions=("dx",),
            gripper_open_command=0.0,
            gripper_closed_command=1.0,
        )


def test_gripper_instruction_does_not_name_motion_complexity_mechanisms():
    interface = ControlInterfaceConfig(
        mode="eef_delta_pose",
        active_dimensions=("dx", "dy", "dz"),
        gripper="direct",
        mapping_strategy="permuted_sign_flipped",
    )
    action_space = ActionSpace.from_config(interface.to_action_space_config())
    instruction = ControlInstructionBuilder().build(
        {**interface.metadata(), "source": "control_interface"},
        action_space.schema(),
    )
    assert "final virtual joystick" in instruction.text
    assert "permuted" not in instruction.text
    assert "sign" not in instruction.text


def test_episode_agent_context_controller_metadata_is_public_only():
    config = load_config(
        "configs/pusher_delta_eef.yaml",
        ("control_interface.mapping_strategy=permuted_sign_flipped", "control_interface.frame_transform=random_rotation_3d"),
    )
    manager = object.__new__(EpisodeManager)
    manager.config = config
    metadata = manager._public_controller_metadata()
    text = str(metadata)
    assert metadata["control_mode"] == "eef_delta_pose"
    assert "active_dimensions" in metadata
    assert "mapping" not in metadata
    assert "mapping_signs" not in text
    assert "mapping_strategy" not in text
    assert "frame_transform" not in text
    assert "rotation_matrix" not in text


def test_prompt_builder_exposes_camera_names_but_not_observation_metadata():
    config = load_config("configs/reacher_delta_eef.yaml")
    action_space = build_action_space(config)
    context = AgentContext(
        observation=Observation(
            step_id=0,
            images={"front": object()},
            state={},
            metadata={"camera_azimuth": "PRIVATE_CAMERA_RANDOMIZATION"},
        ),
        task_instruction="Move to the target.",
        action_space=action_space,
        action_schema=action_space.schema(),
        metadata={"control_interface": control_interface_metadata(config)},
    )

    payload = PromptBuilder().build(context)

    assert payload.camera_names == ("front",)
    assert "PRIVATE_CAMERA_RANDOMIZATION" not in str(payload)
