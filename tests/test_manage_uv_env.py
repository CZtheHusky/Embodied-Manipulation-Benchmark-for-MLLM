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
