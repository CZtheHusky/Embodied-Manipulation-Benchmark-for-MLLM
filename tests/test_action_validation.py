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
