import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config


def _run(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(**{**config.__dict__, "logging": config.logging.__class__(output_dir=tmp_path), "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=123)})
    return BenchmarkRunner(patched).run().episodes[0].metrics


def test_deterministic_seed_behavior(tmp_path):
    metrics_a = _run(tmp_path / "a")
    metrics_b = _run(tmp_path / "b")
    assert metrics_a == metrics_b


def test_selected_episode_id_preserves_derived_seed_and_log_id(tmp_path):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(
        **{
            **config.__dict__,
            "logging": config.logging.__class__(output_dir=tmp_path),
            "benchmark": config.benchmark.__class__(
                name=config.benchmark.name,
                num_episodes=4,
                seed=123,
            ),
        }
    )

    result = BenchmarkRunner(patched).run((2,))

    assert len(result.episodes) == 1
    assert result.episodes[0].episode_id == 2
    assert result.episodes[0].seed == 125
    assert result.episodes[0].log_dir.endswith("episode_0002")


@pytest.mark.parametrize("episode_ids", [(), (2, 2), (-1,), (4,)])
def test_selected_episode_ids_must_be_unique_and_in_configured_range(
    tmp_path,
    episode_ids,
):
    config = load_config("configs/dummy_reach.yaml")
    patched = config.__class__(
        **{
            **config.__dict__,
            "logging": config.logging.__class__(output_dir=tmp_path),
            "benchmark": config.benchmark.__class__(
                name=config.benchmark.name,
                num_episodes=4,
                seed=123,
            ),
        }
    )
    runner = BenchmarkRunner(patched)

    with pytest.raises(ValueError):
        runner.run(episode_ids)
