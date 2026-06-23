import json
from pathlib import Path

from mllm_teleop_benchmark import BenchmarkRunner, load_config


def test_logging_outputs_summary_and_trajectory(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(**{**config.__dict__, "logging": config.logging.__class__(output_dir=tmp_path), "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=config.benchmark.seed)})
    episode = BenchmarkRunner(patched).run().episodes[0]
    episode_dir = Path(episode.log_dir)
    summary = json.loads((episode_dir / "summary.json").read_text())
    assert summary["run_id"] == episode_dir.parent.name
    assert summary["seed"] == config.benchmark.seed
    assert summary["metrics"]["invalid_action_count"] == 0
    assert (episode_dir / "trajectory.jsonl").exists()
    assert (episode_dir / "config.yaml").exists()


def test_repeated_runs_do_not_overwrite_episode_artifacts(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(**{**config.__dict__, "logging": config.logging.__class__(output_dir=tmp_path), "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=config.benchmark.seed)})

    first = BenchmarkRunner(patched).run().episodes[0]
    second = BenchmarkRunner(patched).run().episodes[0]

    first_dir = Path(first.log_dir)
    second_dir = Path(second.log_dir)
    assert first_dir != second_dir
    assert first_dir.parent != second_dir.parent
    assert (first_dir / "summary.json").exists()
    assert (second_dir / "summary.json").exists()
