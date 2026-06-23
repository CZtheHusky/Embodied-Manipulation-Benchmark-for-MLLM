from pathlib import Path

import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config


def test_scripted_policy_succeeds_in_dummy_episode(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(
        **{
            **config.__dict__,
            "logging": config.logging.__class__(output_dir=tmp_path),
            "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=config.benchmark.seed),
        }
    )
    result = BenchmarkRunner(patched).run()
    assert result.aggregate_metrics["success_rate"] == 1.0
    assert result.episodes[0].steps <= config.task.max_steps
    assert Path(result.episodes[0].log_dir).exists()


def test_mllm_placeholder_fails_before_simulator_construction(monkeypatch, tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(
        **{
            **config.__dict__,
            "agent": config.agent.__class__(type="mllm"),
            "logging": config.logging.__class__(output_dir=tmp_path),
            "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=config.benchmark.seed),
        }
    )

    def fail_if_called(_config):
        raise AssertionError("simulator should not be constructed before unsupported mllm agent is rejected")

    monkeypatch.setattr("mllm_teleop_benchmark.benchmark.build_sim", fail_if_called)

    with pytest.raises(ValueError, match="agent.type=mllm is not implemented"):
        BenchmarkRunner(patched).run()
