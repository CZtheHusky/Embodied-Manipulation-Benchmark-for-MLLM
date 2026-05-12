from mllm_teleop_benchmark import BenchmarkRunner, load_config


def _run(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(**{**config.__dict__, "logging": config.logging.__class__(output_dir=tmp_path), "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=123)})
    return BenchmarkRunner(patched).run().episodes[0].metrics


def test_deterministic_seed_behavior(tmp_path):
    metrics_a = _run(tmp_path / "a")
    metrics_b = _run(tmp_path / "b")
    assert metrics_a == metrics_b
