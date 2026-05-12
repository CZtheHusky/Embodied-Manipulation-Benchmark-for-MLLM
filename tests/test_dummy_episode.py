from pathlib import Path

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
