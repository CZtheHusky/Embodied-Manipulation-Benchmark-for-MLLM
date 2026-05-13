from mllm_teleop_benchmark.action.parser import ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.agents.base import AgentContext
from mllm_teleop_benchmark.agents.control_instruction import ControlInstructionBuilder
from mllm_teleop_benchmark.agents.prompt import PromptBuilder
from mllm_teleop_benchmark.config.schema import ControlInterfaceConfig
from mllm_teleop_benchmark.config.loader import load_config
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
    assert "joystick_0 -> dx" in payload.control_instruction
    assert "joystick_1 -> dy" in payload.control_instruction


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


def test_control_instruction_panel_lines_include_mode_mapping_frame_and_scale():
    config = load_config("configs/reacher_delta_eef.yaml")
    action_space = build_action_space(config)
    instruction = ControlInstructionBuilder().build(control_interface_metadata(config), action_space.schema())
    text = "\n".join(instruction.panel_lines)
    assert "control mode: eef_delta_pose" in text
    assert "mapping: fixed" in text
    assert "frame: identity" in text
    assert "scales:" in text


def test_control_interface_frame_transform_override_drives_metadata():
    config = load_config(
        "configs/reacher_delta_eef.yaml",
        ("control_interface.frame_transform=rotation_2d", "control_interface.params.frame_transform.angle_degrees=90"),
    )
    metadata = control_interface_metadata(config)
    assert metadata["frame_transform"] == "rotation_2d"
    assert metadata["frame_transform_metadata"]["type"] == "rotation_2d"
