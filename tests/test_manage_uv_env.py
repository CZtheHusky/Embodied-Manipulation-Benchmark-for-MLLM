import importlib.util
from pathlib import Path

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

    assert {source["name"] for source in simpler_sources} == {"mani-skill2-real2sim", "simpler-env"}
    assert {source["name"] for source in calvin_sources} == {"calvin-env"}
    assert calvin_sources[0]["submodules"] is True
    for source in [*simpler_sources, *calvin_sources]:
        module.validate_source_config(source)


def test_source_config_rejects_branch_or_tag_refs():
    module = load_manage_uv_env()
    source = {
        "name": "floating-source",
        "repo": "https://example.invalid/source.git",
        "commit": "main",
        "install_args": ["--no-deps", "-e"],
        "target_backend": "demo",
    }

    with pytest.raises(SystemExit, match="40-character Git commit SHA"):
        module.validate_source_config(source)


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
        return commit

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
    monkeypatch.setattr(module, "run_output", lambda command, *, cwd=None: commit)

    module.install_source("demo-backend", source)

    source_dir = tmp_path / ".venvs" / "_sources" / "demo-source"
    assert (["git", "submodule", "update", "--init", "--recursive"], source_dir, None) in commands


def test_verify_git_checkout_rejects_mismatched_head(monkeypatch, tmp_path):
    module = load_manage_uv_env()
    expected = "a" * 40
    actual = "b" * 40
    monkeypatch.setattr(module, "run_output", lambda command, *, cwd=None: actual)

    with pytest.raises(SystemExit, match=f"expected {expected}, got {actual}"):
        module.verify_git_checkout(tmp_path, expected)
