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
    with pytest.raises(TypeError):
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
    with pytest.raises(KeyError):
        BenchmarkConfig.from_dict(legacy)


def test_responses_config_uses_full_provider_context():
    config = load_config("configs/reacher_delta_eef_mllm_responses.yaml")

    assert config.agent.type == "mllm"
    assert config.agent.model is not None
    assert config.agent.model.api == "responses"
    assert config.agent.model.reasoning_effort == "low"
    assert config.agent.model.retry_until_success
    assert config.agent.model.retry_backoff_max_seconds == 30
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == "all"
    assert config.agent.history.max_images_per_request == 256
    assert config.agent.history.max_request_bytes == 67_108_864


def test_chat_completions_config_can_use_three_prior_images():
    config = load_config("configs/reacher_delta_eef_mllm_chat_completions.yaml")

    assert config.agent.model is not None
    assert config.agent.model.api == "chat_completions"
    assert config.agent.model.response_format == "json_object"
    assert config.agent.history.action_window == 3
    assert config.agent.history.image_window == 3


def test_responses_backend_accepts_reflection_contract_with_full_history():
    config = load_config(
        "configs/reacher_delta_eef_mllm_chat_completions.yaml",
        (
            "agent.model.api=responses",
            "agent.history.image_window=all",
            "agent.history.action_window=all",
            "agent.history.max_images_per_request=256",
        ),
    )

    assert config.agent.model is not None
    assert config.agent.model.api == "responses"
    assert config.agent.history.action_window == "all"
    assert config.agent.history.image_window == "all"


def test_chat_completions_backend_allows_configurable_paired_history():
    config = load_config(
        "configs/reacher_delta_eef_mllm_chat_completions.yaml",
        (
            "agent.history.image_window=2",
            "agent.history.action_window=2",
        ),
    )

    assert config.agent.history.image_window == 2
    assert config.agent.history.action_window == 2


def test_chat_completions_rejects_unpaired_history_windows():
    with pytest.raises(ValueError, match="action_window to equal"):
        load_config(
            "configs/reacher_delta_eef_mllm_chat_completions.yaml",
            ("agent.history.image_window=2",),
        )


def test_removed_action_output_mode_is_rejected_with_backend_guidance():
    with pytest.raises(ValueError, match="select the backend with agent.model.api"):
        load_config(
            "configs/reacher_delta_eef_mllm_chat_completions.yaml",
            ("agent.action_output_mode=action_with_reflection",),
        )


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ("0", 0),
        ("7", 7),
        ("all", "all"),
    ],
)
def test_mllm_paired_history_override_accepts_zero_integer_or_all(window, expected):
    config = load_config(
        "configs/reacher_delta_eef_mllm_chat_completions.yaml",
        (
            f"agent.history.image_window={window}",
            f"agent.history.action_window={window}",
        ),
    )

    assert config.agent.history.image_window == expected
    assert config.agent.history.action_window == expected


def test_mllm_config_rejects_inline_api_key():
    with pytest.raises(ValueError, match="api_key is forbidden"):
        load_config(
            "configs/reacher_delta_eef_mllm_responses.yaml",
            ("agent.model.api_key=DO_NOT_PERSIST",),
        )


def test_responses_api_requires_explicit_full_history_semantics():
    config = load_config(
        "configs/reacher_delta_eef_mllm_responses.yaml",
        (
            "agent.model.api=responses",
            "agent.history.image_window=all",
            "agent.history.action_window=all",
        ),
    )

    assert config.agent.model is not None
    assert config.agent.model.api == "responses"


@pytest.mark.parametrize("effort", ("none", "low", "medium", "high", "xhigh", "max"))
def test_mllm_config_accepts_supported_reasoning_effort(effort):
    config = load_config(
        "configs/reacher_delta_eef_mllm_responses.yaml",
        (f"agent.model.reasoning_effort={effort}",),
    )

    assert config.agent.model is not None
    assert config.agent.model.reasoning_effort == effort


def test_mllm_config_accepts_thinking_token_budget():
    config = load_config(
        "configs/reacher_delta_eef_mllm_chat_completions.yaml",
        ("agent.model.thinking_token_budget=4096",),
    )

    assert config.agent.model is not None
    assert config.agent.model.thinking_token_budget == 4096


def test_review_video_is_disabled_by_default():
    config = load_config(
        "configs/simpler_env_google_robot_pick_coke_can_mllm_chat_completions.yaml"
    )

    assert not config.logging.save_video
    assert config.logging.video.fps == 2
    assert config.logging.video.step_hold_seconds == 1.0
    assert config.logging.video.terminal_hold_seconds == 2.0
    assert not config.logging.video.include_evaluator_debug


def test_evaluator_debug_video_requires_video_logging():
    with pytest.raises(ValueError, match="requires logging.save_video=true"):
        load_config(
            "configs/reacher_delta_eef_mllm_responses.yaml",
            (
                "logging.save_video=false",
                "logging.video.include_evaluator_debug=true",
            ),
        )


def test_review_video_rejects_non_mllm_agent():
    with pytest.raises(ValueError, match="supports only agent.type=mllm"):
        load_config("configs/dummy_reach.yaml", ("logging.save_video=true",))
