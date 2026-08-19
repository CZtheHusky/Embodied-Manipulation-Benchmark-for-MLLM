from types import SimpleNamespace

import pytest

from mllm_teleop_benchmark import BenchmarkRunner, load_config
from mllm_teleop_benchmark.core.registry import build_sim
from mllm_teleop_benchmark.sim.gymnasium_envs.adapter import GymnasiumSimAdapter


pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")


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


def test_reacher_render_recovers_after_another_egl_context_closes():
    config = load_config("configs/reacher_delta_eef.yaml")
    first = build_sim(config)
    second = build_sim(config)
    second_closed = False
    try:
        first.reset(config.task, seed=9)
        initial = np.asarray(first.render(config.camera.cameras[0]))
        assert initial.shape == (512, 512, 3)
        assert np.any(initial)

        second.reset(config.task, seed=10)
        assert np.any(second.render(config.camera.cameras[0]))
        second.close()
        second_closed = True

        # Closing the current EGL context unbinds it. The surviving simulator
        # must restore its own context and framebuffer before the next read.
        recovered = np.asarray(first.render(config.camera.cameras[0]))
        assert np.any(recovered)
        assert np.array_equal(recovered, initial)
        assert recovered.flags["OWNDATA"]
        assert recovered.flags["C_CONTIGUOUS"]
    finally:
        first.close()
        if not second_closed:
            second.close()


def test_gymnasium_render_rejects_persistent_all_zero_frames(monkeypatch):
    sim = GymnasiumSimAdapter.__new__(GymnasiumSimAdapter)
    sim.config = SimpleNamespace(render_mode="rgb_array")
    sim.env = SimpleNamespace(render=lambda: np.zeros((8, 8, 3), dtype=np.uint8))
    sim.last_render = None
    restore_calls = 0
    recreate_calls = 0

    def restore_context():
        nonlocal restore_calls
        restore_calls += 1

    def recreate_viewer():
        nonlocal recreate_calls
        recreate_calls += 1

    monkeypatch.setattr(sim, "_restore_render_context", restore_context)
    monkeypatch.setattr(sim, "_recreate_render_viewer", recreate_viewer)

    with pytest.raises(RuntimeError, match="all-zero frame"):
        sim.render(SimpleNamespace())

    assert restore_calls == 3
    assert recreate_calls == 1
    assert sim.last_render is None


def test_reacher_render_recreates_viewer_after_persistent_zero_reads(monkeypatch):
    config = load_config("configs/reacher_delta_eef.yaml")
    sim = build_sim(config)
    try:
        sim.reset(config.task, seed=9)
        expected = np.asarray(sim.render(config.camera.cameras[0]))
        assert np.any(expected)

        original_render = sim.env.render
        render_calls = 0

        def two_zero_reads_then_render():
            nonlocal render_calls
            render_calls += 1
            if render_calls <= 2:
                return np.zeros_like(expected)
            return original_render()

        monkeypatch.setattr(sim.env, "render", two_zero_reads_then_render)
        recovered = np.asarray(sim.render(config.camera.cameras[0]))

        assert render_calls == 3
        assert np.any(recovered)
        assert np.array_equal(recovered, expected)
    finally:
        sim.close()
