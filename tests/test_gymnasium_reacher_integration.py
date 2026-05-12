import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config


pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
pytest.importorskip("numpy")


def test_reacher_delta_eef_env_short_episode(tmp_path):
    config = load_config("configs/reacher_delta_eef.yaml")
    patched = config.__class__(
        **{
            **config.__dict__,
            "logging": config.logging.__class__(output_dir=tmp_path),
            "benchmark": config.benchmark.__class__(name=config.benchmark.name, num_episodes=1, seed=11),
            "task": config.task.__class__(
                type=config.task.type,
                max_steps=3,
                success_threshold=config.task.success_threshold,
                target_position=config.task.target_position,
                initial_ee_position=config.task.initial_ee_position,
                goal_hold_steps=config.task.goal_hold_steps,
                params=config.task.params,
            ),
        }
    )
    result = BenchmarkRunner(patched).run()
    assert result.aggregate_metrics["num_episodes"] == 1
    assert result.episodes[0].steps <= 3
    assert "final_distance_to_target" in result.episodes[0].metrics
