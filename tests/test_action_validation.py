import pytest

from mllm_teleop_benchmark.action.parser import ActionParseError, ActionParser
from mllm_teleop_benchmark.action.space import ActionSpace
from mllm_teleop_benchmark.action.validator import ActionValidator
from mllm_teleop_benchmark.config.loader import load_config


def _validator():
    config = load_config("configs/dummy_reach.yaml")
    return ActionValidator(ActionSpace.from_config(config.action_space))


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


def test_invalid_json_fails():
    with pytest.raises(ActionParseError):
        ActionParser().parse("{bad json")


def test_action_only_model_output_rejects_top_level_reflection():
    with pytest.raises(ActionParseError, match="Unknown top-level"):
        ActionParser().parse_model_output(
            {
                "channels": {f"joystick_{idx}": 0 for idx in range(7)},
                "Reflection": "joystick_0 may move left",
            },
            "action_only",
        )


def test_reflection_model_output_preserves_optional_string():
    parsed = ActionParser().parse_model_output(
        {
            "channels": {f"joystick_{idx}": 0 for idx in range(7)},
            "Reflection": "joystick_0 may move left",
        },
        "action_with_reflection",
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
            "action_with_reflection",
        )


def test_model_output_requires_explicit_channels_wrapper():
    with pytest.raises(ActionParseError, match="top-level 'channels'"):
        ActionParser().parse_model_output(
            {f"joystick_{idx}": 0 for idx in range(7)},
            "action_only",
        )
