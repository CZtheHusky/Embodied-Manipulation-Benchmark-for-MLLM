import json
from pathlib import Path

from mllm_teleop_benchmark import BenchmarkRunner, load_config


def test_logging_outputs_summary_and_trajectory(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(**{**config.__dict__, "logging": config.logging.__class__(output_dir=tmp_path), "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=config.benchmark.seed)})
    episode = BenchmarkRunner(patched).run().episodes[0]
    episode_dir = Path(episode.log_dir)
    summary = json.loads((episode_dir / "summary.json").read_text())
    assert summary["seed"] == config.benchmark.seed
    assert summary["metrics"]["invalid_action_count"] == 0
    assert (episode_dir / "trajectory.jsonl").exists()
    assert (episode_dir / "config.yaml").exists()
