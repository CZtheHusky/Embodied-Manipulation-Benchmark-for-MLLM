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


def test_cli_style_override_updates_ui_config():
    config = load_config(
        "configs/reacher_delta_eef.yaml",
        ("ui.backend=pygame", "ui.runtime=sim_adapter_worker", "ui.worker_timeout_seconds=3"),
    )

    assert config.ui.backend == "pygame"
    assert config.ui.runtime == "sim_adapter_worker"
    assert config.ui.worker_timeout_seconds == 3.0


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


def test_mllm_config_uses_full_action_history_and_two_prior_images_by_default():
    config = load_config("configs/reacher_delta_eef_mllm.yaml")

    assert config.agent.type == "mllm"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == 2
    assert config.agent.history.max_images_per_request == 128
    assert config.agent.history.max_request_bytes == 67_108_864


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("agent.history.image_window=0", 0),
        ("agent.history.image_window=7", 7),
        ("agent.history.image_window=all", "all"),
    ],
)
def test_mllm_image_history_override_accepts_zero_integer_or_all(override, expected):
    config = load_config("configs/reacher_delta_eef_mllm.yaml", (override,))

    assert config.agent.history.image_window == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "true", "recent"])
def test_mllm_image_history_rejects_ambiguous_or_negative_values(value):
    with pytest.raises(ValueError, match="non-negative integer or 'all'"):
        load_config(
            "configs/reacher_delta_eef_mllm.yaml",
            (f"agent.history.image_window={value}",),
        )


def test_mllm_config_rejects_inline_api_key():
    with pytest.raises(ValueError, match="api_key is forbidden"):
        load_config(
            "configs/reacher_delta_eef_mllm.yaml",
            ("agent.model.api_key=DO_NOT_PERSIST",),
        )


def test_review_video_is_disabled_by_default():
    config = load_config("configs/reacher_delta_eef_mllm.yaml")

    assert not config.logging.save_video
    assert config.logging.video.fps == 2
    assert config.logging.video.step_hold_seconds == 1.0
    assert config.logging.video.terminal_hold_seconds == 2.0
    assert not config.logging.video.include_evaluator_debug


def test_evaluator_debug_video_requires_video_logging():
    with pytest.raises(ValueError, match="requires logging.save_video=true"):
        load_config(
            "configs/reacher_delta_eef_mllm.yaml",
            ("logging.video.include_evaluator_debug=true",),
        )


def test_review_video_rejects_non_mllm_agent():
    with pytest.raises(ValueError, match="supports only agent.type=mllm"):
        load_config("configs/dummy_reach.yaml", ("logging.save_video=true",))


@pytest.mark.parametrize(
    "override",
    (
        "logging.video.fps=0",
        "logging.video.step_hold_seconds=0",
        "logging.video.terminal_hold_seconds=-1",
    ),
)
def test_review_video_rejects_non_positive_timing(override):
    with pytest.raises(ValueError, match="must be positive"):
        load_config("configs/reacher_delta_eef_mllm.yaml", (override,))
