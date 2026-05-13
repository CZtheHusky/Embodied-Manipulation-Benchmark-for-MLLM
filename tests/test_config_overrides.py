from mllm_teleop_benchmark.config.loader import load_config
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
