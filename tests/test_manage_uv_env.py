import importlib.util
from pathlib import Path
import re

import pytest


def load_manage_uv_env():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "manage_uv_env.py"
    spec = importlib.util.spec_from_file_location("manage_uv_env", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_backends_reports_support_levels():
    module = load_manage_uv_env()

    lines = module.list_backends()

    assert [line.split("\t")[0] for line in lines] == list(module.BACKENDS)
    assert "core\tuv" in lines
    assert "metaworld\tuv" in lines
    assert "maniskill-cu126\texternal_runtime" in lines
    assert "simpler-env\tuv_source" in lines
    assert "calvin\tuv_source" in lines
    assert "language-table\tuv_source" in lines


def test_robocasa_runner_enforces_offscreen_egl(monkeypatch):
    module = load_manage_uv_env()
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "glx")

    env = module.backend_env("robocasa")

    assert env["MUJOCO_GL"] == "egl"
    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert env["NUMBA_DISABLE_JIT"] == "1"


def test_blocked_backend_fails_before_uv_commands():
    module = load_manage_uv_env()
    module.BACKENDS["blocked-sim"] = {
        "python": "3.11",
        "input": "blocked-sim.in",
        "support_level": "fallback_blocked",
        "blocked_reason": "requires a conda or container fallback",
        "doc": "docs/simulator_backend_status.md",
    }

    with pytest.raises(SystemExit) as excinfo:
        module.lock_backend("blocked-sim", module.PYTHON_PLATFORM)

    message = str(excinfo.value)
    assert "blocked-sim uv lock is blocked" in message
    assert "requires a conda or container fallback" in message
    assert "docs/simulator_backend_status.md" in message


def test_uv_source_backends_declare_exact_source_commits():
    module = load_manage_uv_env()

    simpler_sources = module.backend_sources("simpler-env")
    calvin_sources = module.backend_sources("calvin")
    language_table_sources = module.backend_sources("language-table")

    assert {source["name"] for source in simpler_sources} == {"mani-skill2-real2sim", "simpler-env"}
    assert {source["name"] for source in calvin_sources} == {"calvin-env"}
    assert {source["name"] for source in language_table_sources} == {"language-table"}
    assert calvin_sources[0]["submodules"] is True
    assert language_table_sources[0]["install"] is False
    assert language_table_sources[0]["pythonpath"] is True
    for source in [*simpler_sources, *calvin_sources, *language_table_sources]:
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])


def test_simpler_env_smoke_includes_mllm_integration_contract():
    module = load_manage_uv_env()

    smoke_commands = module.BACKENDS["simpler-env"]["smoke"]

    assert len(smoke_commands) == 1
    assert "tests/test_simpler_env_backend.py" in smoke_commands[0]
    assert "tests/test_simpler_env_integration.py" in smoke_commands[0]


def test_install_source_checks_out_and_installs_pinned_commit(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    commit = "a" * 40
    source = {
        "name": "demo-source",
        "repo": "https://example.invalid/demo-source.git",
        "commit": commit,
        "install_args": ["--no-deps", "-e"],
        "target_backend": "demo-backend",
    }
    commands = []

    def fake_run(command, *, cwd=None, env=None):
        commands.append((command, cwd, env))

    def fake_run_output(command, *, cwd=None):
        commands.append((command, cwd, None))
        return commit if command[:2] == ["git", "rev-parse"] else ""

    monkeypatch.setattr(module, "VENV_DIR", tmp_path / ".venvs")
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "run_output", fake_run_output)

    module.install_source("demo-backend", source)

    source_dir = tmp_path / ".venvs" / "_sources" / "demo-source"
    assert (["git", "fetch", "--depth", "1", "origin", commit], source_dir, None) in commands
    assert (["git", "checkout", "--detach", commit], source_dir, None) in commands
    assert (["git", "rev-parse", "HEAD"], source_dir, None) in commands
    assert commands[-1] == (
        ["uv", "pip", "install", "--python", str(tmp_path / ".venvs" / "demo-backend"), "--no-deps", "-e", str(source_dir)],
        None,
        None,
    )


def test_install_source_updates_submodules_when_requested(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    commit = "a" * 40
    source = {
        "name": "demo-source",
        "repo": "https://example.invalid/demo-source.git",
        "commit": commit,
        "install_args": ["--no-deps", "-e"],
        "target_backend": "demo-backend",
        "submodules": True,
    }
    commands = []

    monkeypatch.setattr(module, "VENV_DIR", tmp_path / ".venvs")
    monkeypatch.setattr(module, "run", lambda command, *, cwd=None, env=None: commands.append((command, cwd, env)))
    monkeypatch.setattr(
        module,
        "run_output",
        lambda command, *, cwd=None: (
            commit if command[:2] == ["git", "rev-parse"] else ""
        ),
    )

    module.install_source("demo-backend", source)

    source_dir = tmp_path / ".venvs" / "_sources" / "demo-source"
    assert (["git", "submodule", "update", "--init", "--recursive"], source_dir, None) in commands


def test_source_only_checkout_is_not_passed_to_pip(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    commit = "a" * 40
    source = {
        "name": "source-only",
        "repo": "https://example.invalid/source-only.git",
        "commit": commit,
        "install": False,
        "pythonpath": True,
    }
    commands = []

    monkeypatch.setattr(module, "VENV_DIR", tmp_path / ".venvs")
    monkeypatch.setattr(
        module,
        "run",
        lambda command, *, cwd=None, env=None: commands.append((command, cwd, env)),
    )
    monkeypatch.setattr(
        module,
        "run_output",
        lambda command, *, cwd=None: (
            commit if command[:2] == ["git", "rev-parse"] else ""
        ),
    )

    module.install_source("demo-backend", source)

    assert not any(command[0][:3] == ["uv", "pip", "install"] for command in commands)


def test_language_table_source_is_added_to_pythonpath(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    monkeypatch.setattr(module, "VENV_DIR", tmp_path / ".venvs")

    env = module.backend_env("language-table")

    assert env["PYTHONPATH"].split(module.os.pathsep) == [
        str(module.ROOT / "src"),
        str(tmp_path / ".venvs" / "_sources" / "language-table"),
    ]


def test_verify_git_checkout_rejects_mismatched_head(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    expected = "a" * 40
    actual = "b" * 40
    monkeypatch.setattr(module, "run_output", lambda command, *, cwd=None: actual)

    with pytest.raises(SystemExit, match=f"expected {expected}, got {actual}"):
        module.verify_git_checkout(tmp_path, expected)


def test_verify_git_checkout_rejects_tracked_modifications(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    expected = "a" * 40

    def fake_run_output(command, *, cwd=None):
        if command[:2] == ["git", "rev-parse"]:
            return expected
        return " M simulator.py"

    monkeypatch.setattr(module, "run_output", fake_run_output)

    with pytest.raises(SystemExit, match="tracked modifications"):
        module.verify_git_checkout(tmp_path, expected)


def test_run_backend_verifies_sources_before_command(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    python = tmp_path / "python"
    python.touch()
    calls = []
    monkeypatch.setattr(module, "python_bin", lambda _name: python)
    monkeypatch.setattr(module, "verify_backend_sources", lambda name: calls.append(("verify", name)))
    monkeypatch.setattr(
        module,
        "run",
        lambda command, *, cwd=None, env=None: calls.append(("run", command)),
    )

    module.run_backend("simpler-env", ["python", "-c", "pass"])

    assert calls == [
        ("verify", "simpler-env"),
        ("run", [str(python), "-c", "pass"]),
    ]
