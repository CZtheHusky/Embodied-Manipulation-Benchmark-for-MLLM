from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from mllm_teleop_benchmark.config.loader import load_config
from mllm_teleop_benchmark.sim.colosseum.adapter import ColosseumSimAdapter
from mllm_teleop_benchmark.sim.coppeliasim_lock import (
    COPPELIASIM_LOCK_PATH_ENV,
    DEFAULT_COPPELIASIM_LOCK_PATH,
    CoppeliaSimRuntimeBusyError,
    CoppeliaSimRuntimeLock,
)
from mllm_teleop_benchmark.sim.rlbench.adapter import RLBenchSimAdapter


ROOT = Path(__file__).resolve().parents[1]
RLBENCH_CONFIG = "configs/rlbench_reach_target_delta_eef.yaml"
COLOSSEUM_CONFIG = "configs/colosseum_close_box_manip_obj_color_smoke.yaml"


class FakeEnvironment:
    def __init__(
        self,
        launch_error: Exception | None = None,
        shutdown_error: Exception | None = None,
    ) -> None:
        self.launch_error = launch_error
        self.shutdown_error = shutdown_error
        self.launch_calls = 0
        self.shutdown_calls = 0

    def launch(self) -> None:
        self.launch_calls += 1
        if self.launch_error is not None:
            raise self.launch_error

    def get_task(self, task_class):
        return task_class()

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _adapters():
    rlbench = RLBenchSimAdapter(load_config(RLBENCH_CONFIG).simulator)
    colosseum = ColosseumSimAdapter(load_config(COLOSSEUM_CONFIG).simulator)
    return rlbench, colosseum


def test_rlbench_and_colosseum_instances_share_lock_until_close(
    monkeypatch,
    tmp_path,
):
    # WHY: backend-specific locks would still let both adapters launch the same
    # host CoppeliaSim runtime and corrupt one another's simulator session.
    lock_path = tmp_path / "shared-runtime.lock"
    monkeypatch.setenv(COPPELIASIM_LOCK_PATH_ENV, str(lock_path))
    rlbench, colosseum = _adapters()
    rlbench_env = FakeEnvironment()
    blocked_env = FakeEnvironment()
    rlbench.env = rlbench_env
    colosseum.env = blocked_env

    rlbench._launch_environment(object)

    with pytest.raises(CoppeliaSimRuntimeBusyError) as exc_info:
        colosseum._launch_environment(object)

    assert "another RLBench or Colosseum process" in str(exc_info.value)
    assert str(lock_path) in str(exc_info.value)
    assert blocked_env.launch_calls == 0
    assert blocked_env.shutdown_calls == 1

    rlbench.close()
    colosseum_env = FakeEnvironment()
    colosseum.env = colosseum_env
    colosseum._launch_environment(object)
    colosseum.close()
    colosseum.close()

    assert rlbench_env.shutdown_calls == 1
    assert colosseum_env.shutdown_calls == 1


@pytest.mark.parametrize("failing_backend", ["rlbench", "colosseum"])
def test_initialization_failure_releases_lock(
    failing_backend,
    monkeypatch,
    tmp_path,
):
    # WHY: a failed launch must not strand every later benchmark run behind a
    # lock whose simulator never became usable.
    monkeypatch.setenv(
        COPPELIASIM_LOCK_PATH_ENV,
        str(tmp_path / "failed-launch.lock"),
    )
    rlbench, colosseum = _adapters()
    failing = rlbench if failing_backend == "rlbench" else colosseum
    contender = colosseum if failing_backend == "rlbench" else rlbench
    failed_env = FakeEnvironment(RuntimeError("synthetic launch failure"))
    failing.env = failed_env

    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        failing._launch_environment(object)

    contender_env = FakeEnvironment()
    contender.env = contender_env
    contender._launch_environment(object)
    contender.close()

    assert failed_env.shutdown_calls == 1
    assert contender_env.launch_calls == 1


def test_close_releases_lock_when_shutdown_fails(
    monkeypatch,
    tmp_path,
):
    # WHY: simulator teardown errors must not strand the host lock or make a
    # later idempotent close repeat a partially completed shutdown.
    monkeypatch.setenv(
        COPPELIASIM_LOCK_PATH_ENV,
        str(tmp_path / "failed-shutdown.lock"),
    )
    rlbench, colosseum = _adapters()
    rlbench_env = FakeEnvironment(
        shutdown_error=RuntimeError("synthetic shutdown failure")
    )
    rlbench.env = rlbench_env
    rlbench._launch_environment(object)

    with pytest.raises(RuntimeError, match="synthetic shutdown failure"):
        rlbench.close()
    rlbench.close()

    contender_env = FakeEnvironment()
    colosseum.env = contender_env
    colosseum._launch_environment(object)
    colosseum.close()

    assert rlbench_env.shutdown_calls == 1
    assert contender_env.launch_calls == 1


def test_second_process_fails_without_waiting_for_host_lock(
    monkeypatch,
    tmp_path,
):
    # WHY: independently scheduled benchmark workers must reject the collision
    # before launch rather than block indefinitely or race CoppeliaSim startup.
    lock_path = tmp_path / "cross-process.lock"
    monkeypatch.setenv(COPPELIASIM_LOCK_PATH_ENV, str(lock_path))
    holder = CoppeliaSimRuntimeLock("rlbench")
    holder.acquire()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(ROOT / "src"), env.get("PYTHONPATH")),
        )
    )
    script = """
import sys
from mllm_teleop_benchmark.sim.coppeliasim_lock import (
    CoppeliaSimRuntimeBusyError,
    CoppeliaSimRuntimeLock,
)

try:
    CoppeliaSimRuntimeLock("colosseum").acquire()
except CoppeliaSimRuntimeBusyError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(23)
raise SystemExit(0)
"""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    finally:
        holder.release()

    assert completed.returncode == 23
    assert "CoppeliaSim v4.1 is already in use" in completed.stderr
    assert str(lock_path) in completed.stderr


def test_lock_ignores_file_contents(monkeypatch, tmp_path):
    # WHY: stale or untrusted text in /tmp is not proof that a process owns the
    # runtime; only the kernel-maintained advisory lock determines ownership.
    lock_path = tmp_path / "content-is-not-ownership.lock"
    lock_path.write_text("not trusted as ownership metadata", encoding="utf-8")
    monkeypatch.setenv(COPPELIASIM_LOCK_PATH_ENV, str(lock_path))
    runtime_lock = CoppeliaSimRuntimeLock("rlbench")

    runtime_lock.acquire()
    runtime_lock.release()

    assert lock_path.read_text(encoding="utf-8") == (
        "not trusted as ownership metadata"
    )
    assert DEFAULT_COPPELIASIM_LOCK_PATH == Path(
        "/tmp/mllm-teleop-coppeliasim-v4.1.lock"
    )
