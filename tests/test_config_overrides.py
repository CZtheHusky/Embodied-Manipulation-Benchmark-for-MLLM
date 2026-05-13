import pytest

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.registry import control_interface_metadata


def test_cli_style_override_updates_control_interface():
    config = load_config(
        "configs/reacher_delta_eef.yaml",
        ("control_interface.mapping_strategy=permuted_sign_flipped", "control_interface.mapping_visible_to_agent=false"),
    )
    assert config.control_interface is not None
    assert config.control_interface.mapping_strategy == "permuted_sign_flipped"
    assert not config.control_interface.mapping_visible_to_agent
    metadata = control_interface_metadata(config)
    assert "mapping" not in metadata


def test_control_interface_is_required():
    with pytest.raises(ValueError, match="control_interface"):
        load_config("configs/dummy_reach.yaml", ("control_interface=null",))


def test_legacy_action_space_mapping_without_control_interface_is_rejected():
    legacy = {
        "benchmark": {"name": "legacy", "num_episodes": 1, "seed": 1},
        "simulator": {"type": "dummy"},
        "task": {"type": "reach_target", "max_steps": 1, "success_threshold": 0.1},
        "camera": {"visible_metadata": True, "cameras": [{"name": "front"}]},
        "action_space": {"type": "joystick", "channels": {"joystick_0": {"kind": "discrete", "values": [-1, 0, 1]}}},
        "mapping": {"type": "fixed", "visible_to_agent": True, "channels": {"joystick_0": "dx"}},
        "controller": {"type": "gymnasium_box", "scales": {}, "params": {"dimensions": ["dx"]}},
        "frame_transform": {"type": "identity", "visible_to_agent": True},
        "agent": {"type": "random"},
        "logging": {"output_dir": "outputs/legacy"},
    }
    with pytest.raises(ValueError, match="control_interface"):
        BenchmarkConfig.from_dict(legacy)
