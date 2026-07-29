import importlib.util
from pathlib import Path

import pytest


def load_manage_conda_env():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "manage_conda_env.py"
    )
    spec = importlib.util.spec_from_file_location("manage_conda_env", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_git_checkout_accepts_exact_clean_commit(monkeypatch, tmp_path):
    module = load_manage_conda_env()
    expected = "a" * 40

    def fake_run_output(command, *, cwd=None):
        if command[:2] == ["git", "rev-parse"]:
            return expected
        return ""

    monkeypatch.setattr(module, "run_output", fake_run_output)

    module.verify_git_checkout(tmp_path, expected)


def test_verify_git_checkout_rejects_tracked_modifications(monkeypatch, tmp_path):
    module = load_manage_conda_env()
    expected = "a" * 40

    def fake_run_output(command, *, cwd=None):
        if command[:2] == ["git", "rev-parse"]:
            return expected
        return " M simulator.py"

    monkeypatch.setattr(module, "run_output", fake_run_output)

    with pytest.raises(SystemExit, match="tracked modifications"):
        module.verify_git_checkout(tmp_path, expected)


def test_run_backend_verifies_sources_before_conda_command(monkeypatch):
    module = load_manage_conda_env()
    calls = []
    monkeypatch.setattr(module, "verify_backend_sources", lambda name: calls.append(("verify", name)))
    monkeypatch.setattr(
        module,
        "run_conda",
        lambda name, command, *, env=None: calls.append(("run", name, command)),
    )

    module.run_backend("rlbench", ["python", "-c", "pass"])

    assert calls == [
        ("verify", "rlbench"),
        ("run", "rlbench", ["python", "-c", "pass"]),
    ]
