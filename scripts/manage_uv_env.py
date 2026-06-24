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
    "simpler-env": {
        "python": "3.10",
        "input": "simpler-env.in",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_simpler_env_backend.py", "-rs"]],
    },
    "calvin": {
        "python": "3.10",
        "input": "calvin.in",
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_calvin_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/calvin_delta_eef.yaml"],
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


def run_backend(name: str, command: list[str]) -> None:
    python = python_bin(name)
    if not python.exists():
        raise SystemExit(f"uv environment does not exist: {venv_path(name)}; run create first")
    if command[0] == "python":
        command = [str(python), *command[1:]]
    elif command[0].endswith("/python"):
        command[0] = str(python)
    run(command, env=backend_env(name))


def smoke_backend(name: str) -> None:
    cfg = backend_config(name)
    for command in cfg["smoke"]:  # type: ignore[index]
        run_backend(name, list(command))


def backend_env(name: str) -> dict[str, str]:
    cfg = backend_config(name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
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
