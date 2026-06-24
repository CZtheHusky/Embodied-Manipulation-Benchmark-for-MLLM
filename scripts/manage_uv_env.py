#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
UV_DIR = ROOT / "envs" / "uv"
LOCK_DIR = UV_DIR / "locks"
VENV_DIR = ROOT / ".venvs"
PYTHON_PLATFORM = "x86_64-unknown-linux-gnu"
CALVIN_REPO = "https://github.com/mees/calvin_env.git"
CALVIN_COMMIT = "797142c588c21e76717268b7b430958dbd13bf48"
CUDA_SMOKE = (
    "import torch; "
    "assert torch.cuda.is_available(), 'CUDA is not available'; "
    "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
)
LIBERO_UV_BLOCKED = (
    "LIBERO uv setup is blocked and not runtime verified. The Python 3.10/cu113 lock attempt "
    "does not complete when the pinned upstream LIBERO Git source is included, so no "
    "envs/uv/locks/libero.lock.txt is checked in. Use the conda fallback documented in "
    "docs/libero_integration.md."
)


BACKENDS: dict[str, dict[str, object]] = {
    "core": {
        "python": "3.11",
        "input": "core.in",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_dummy_episode.py", "tests/test_control_interface.py", "tests/test_mapping.py", "tests/test_frame_transform.py"]],
    },
    "gymnasium": {
        "python": "3.11",
        "input": "gymnasium.in",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_gymnasium_controller.py", "tests/test_dummy_episode.py"]],
    },
    "dmcontrol": {
        "python": "3.11",
        "input": "dmcontrol.in",
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_dmcontrol_integration.py"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/dmcontrol_example.yaml"],
        ],
    },
    "robosuite": {
        "python": "3.10",
        "input": "robosuite.in",
        "env": {"NUMBA_DISABLE_JIT": "1"},
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_robosuite_integration.py"]],
    },
    "maniskill-cpu": {
        "python": "3.11",
        "input": "maniskill-cpu.in",
        "torch_backend": "cpu",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_maniskill_backend.py"]],
    },
    "maniskill-cu126": {
        "python": "3.11",
        "input": "maniskill-cu126.in",
        "torch_backend": "cu126",
        "smoke": [
            ["python", "-c", CUDA_SMOKE],
            ["python", "-m", "pytest", "-q", "tests/test_maniskill_backend.py"],
        ],
    },
    "calvin": {
        "python": "3.10",
        "input": "calvin.in",
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_calvin_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/calvin_delta_eef.yaml"],
        ],
    },
    "libero": {
        "python": "3.10",
        "input": "libero.in",
        "torch_backend": "cu113",
        "blocked_reason": LIBERO_UV_BLOCKED,
        "env": {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "NUMBA_DISABLE_JIT": "1"},
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_libero_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/libero_spatial_task0.yaml"],
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage uv environments for simulator backends.")
    parser.add_argument("command", choices=("lock", "create", "run", "smoke", "list"))
    parser.add_argument("backend", nargs="?", choices=tuple(BACKENDS))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Command to pass after -- for run.")
    parser.add_argument("--platform", default=PYTHON_PLATFORM)
    ns = parser.parse_args()

    if ns.command == "list":
        for name in BACKENDS:
            print(name)
        return
    if ns.backend is None:
        parser.error("backend is required")

    if ns.command == "lock":
        lock_backend(ns.backend, ns.platform)
    elif ns.command == "create":
        create_backend(ns.backend)
    elif ns.command == "run":
        command = list(ns.args)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("run requires a command after --")
        run_backend(ns.backend, command)
    elif ns.command == "smoke":
        smoke_backend(ns.backend)


def backend_config(name: str) -> dict[str, object]:
    return BACKENDS[name]


def input_path(name: str) -> Path:
    return UV_DIR / str(backend_config(name)["input"])


def lock_path(name: str) -> Path:
    return LOCK_DIR / f"{name}.lock.txt"


def venv_path(name: str) -> Path:
    return VENV_DIR / name


def python_bin(name: str) -> Path:
    return venv_path(name) / "bin" / "python"


def lock_backend(name: str, platform: str) -> None:
    require_unblocked_backend(name, "lock")
    cfg = backend_config(name)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "pip",
        "compile",
        str(input_path(name)),
        "--python-version",
        str(cfg["python"]),
        "--python-platform",
        platform,
        "--output-file",
        str(lock_path(name)),
    ]
    torch_backend = cfg.get("torch_backend")
    if torch_backend:
        command.extend(["--torch-backend", str(torch_backend)])
    run(command)


def create_backend(name: str) -> None:
    require_unblocked_backend(name, "create")
    cfg = backend_config(name)
    if not lock_path(name).exists():
        lock_backend(name, PYTHON_PLATFORM)
    VENV_DIR.mkdir(parents=True, exist_ok=True)
    run(["uv", "venv", str(venv_path(name)), "--python", str(cfg["python"]), "--clear"])
    sync_command = ["uv", "pip", "sync", "--python", str(venv_path(name)), str(lock_path(name))]
    torch_backend = cfg.get("torch_backend")
    if torch_backend:
        sync_command.extend(["--torch-backend", str(torch_backend)])
    run(sync_command)
    run(["uv", "pip", "install", "--python", str(venv_path(name)), "-e", str(ROOT)])
    if name == "calvin":
        install_calvin_editable()
    if name == "libero":
        write_libero_config()


def install_calvin_editable() -> None:
    source_dir = VENV_DIR / "_sources" / "calvin-env"
    if source_dir.exists():
        run(["git", "fetch", "origin"], cwd=source_dir)
    else:
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", CALVIN_REPO, str(source_dir)])
    run(["git", "checkout", CALVIN_COMMIT], cwd=source_dir)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)
    run(["uv", "pip", "install", "--python", str(venv_path("calvin")), "-e", str(source_dir)])


def write_libero_config() -> None:
    config_dir = VENV_DIR / "libero_config"
    script = (
        "from pathlib import Path; "
        "import importlib.util, yaml; "
        "spec = importlib.util.find_spec('libero.libero'); "
        "assert spec and spec.origin, 'libero.libero is not importable'; "
        "root = Path(spec.origin).resolve().parent; "
        f"cfg_dir = Path({str(config_dir)!r}); "
        "cfg_dir.mkdir(parents=True, exist_ok=True); "
        "yaml.safe_dump({"
        "'benchmark_root': str(root), "
        "'bddl_files': str(root / 'bddl_files'), "
        "'init_states': str(root / 'init_files'), "
        "'datasets': str(root.parent / 'datasets'), "
        "'assets': str(root / 'assets')"
        "}, (cfg_dir / 'config.yaml').open('w', encoding='utf-8'), sort_keys=True)"
    )
    run([str(python_bin("libero")), "-c", script], env=backend_env("libero"))


def run_backend(name: str, command: list[str]) -> None:
    require_unblocked_backend(name, "run")
    python = python_bin(name)
    if not python.exists():
        raise SystemExit(f"uv environment does not exist: {venv_path(name)}; run create first")
    if command[0] == "python":
        command = [str(python), *command[1:]]
    elif command[0].endswith("/python"):
        command[0] = str(python)
    run(command, env=backend_env(name))


def smoke_backend(name: str) -> None:
    require_unblocked_backend(name, "smoke")
    cfg = backend_config(name)
    for command in cfg["smoke"]:  # type: ignore[index]
        run_backend(name, list(command))


def require_unblocked_backend(name: str, command: str) -> None:
    reason = backend_config(name).get("blocked_reason")
    if reason:
        raise SystemExit(f"{name} uv {command} is blocked: {reason}")


def backend_env(name: str) -> dict[str, str]:
    cfg = backend_config(name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    if name == "libero":
        env.setdefault("LIBERO_CONFIG_PATH", str(VENV_DIR / "libero_config"))
    env.update(cfg.get("env", {}))  # type: ignore[arg-type]
    return env


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd or ROOT), env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
