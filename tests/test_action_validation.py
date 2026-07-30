import pytest

from mllm_teleop_benchmark.action.parser import ActionParseError, ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import ControlInterfaceConfig


def _validator():
    config = load_config("configs/dummy_reach.yaml")
    return ActionValidator(ActionSpace.from_config(config.action_space))


def _continuous_motion_space(*, gripper="none"):
    interface = ControlInterfaceConfig(
        mode="eef_delta_pose",
        active_dimensions=("dx", "dy"),
        gripper=gripper,
        channel_kind="continuous",
        channel_minimum=-30.0,
        channel_maximum=30.0,
    )
    return interface, ActionSpace.from_config(interface.to_action_space_config())


def test_valid_action_passes():
    action = ActionParser().parse({f"joystick_{idx}": 0 for idx in range(7)})
    assert _validator().validate(action).is_valid


def test_discrete_channel_rejects_continuous_value():
    action = ActionParser().parse({**{f"joystick_{idx}": 0 for idx in range(7)}, "joystick_0": 0.5})
    result = _validator().validate(action)
    assert not result.is_valid
    assert "joystick_0" in result.errors[0]


def test_gripper_range_validation():
    action = ActionParser().parse({**{f"joystick_{idx}": 0 for idx in range(7)}, "joystick_6": 1.5})
    result = _validator().validate(action)
    assert not result.is_valid


def test_continuous_model_motion_accepts_fractional_values():
    _, action_space = _continuous_motion_space()
    action = ActionParser().parse(
        {"joystick_0": 12.5, "joystick_1": -29.75}
    )

    assert ActionValidator(action_space).validate(action).is_valid


def test_continuous_model_motion_clips_only_selected_channels():
    interface, action_space = _continuous_motion_space(gripper="direct")
    action = ActionParser().parse(
        {"joystick_0": 45, "joystick_1": -31.5, "joystick_2": 1.5}
    )

    clipped = action_space.clip_continuous_channels(
        action,
        interface.channel_names()[: len(interface.active_dimensions)],
    )
    result = ActionValidator(action_space).validate(clipped)

    assert clipped.channels == {
        "joystick_0": 30.0,
        "joystick_1": -30.0,
        "joystick_2": 1.5,
    }
    assert not result.is_valid
    assert "joystick_2" in result.errors[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_model_motion_is_not_converted_to_a_boundary(value):
    interface, action_space = _continuous_motion_space()
    action = ActionParser().parse(
        {"joystick_0": value, "joystick_1": 0}
    )

    clipped = action_space.clip_continuous_channels(
        action,
        interface.channel_names(),
    )

    assert not ActionValidator(action_space).validate(clipped).is_valid


def test_invalid_json_fails():
    with pytest.raises(ActionParseError):
        ActionParser().parse("{bad json")


def test_model_output_preserves_optional_reflection_string():
    parsed = ActionParser().parse_model_output(
        {
            "channels": {f"joystick_{idx}": 0 for idx in range(7)},
            "Reflection": "joystick_0 may move left",
        },
    )

    assert parsed.action.channels["joystick_0"] == 0
    assert parsed.reflection == "joystick_0 may move left"


def test_reflection_model_output_rejects_non_string_reflection():
    with pytest.raises(ActionParseError, match="must be a string"):
        ActionParser().parse_model_output(
            {
                "channels": {f"joystick_{idx}": 0 for idx in range(7)},
                "Reflection": {"joystick_0": "left"},
            },
        )


def test_model_output_requires_explicit_channels_wrapper():
    with pytest.raises(ActionParseError, match="top-level 'channels'"):
        ActionParser().parse_model_output(
            {f"joystick_{idx}": 0 for idx in range(7)},
        )
