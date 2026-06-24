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
CUDA_SMOKE = (
    "import torch; "
    "assert torch.cuda.is_available(), 'CUDA is not available'; "
    "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
)
RLBENCH_UV_BLOCKED = (
    "RLBench uv setup is blocked and not runtime verified. The pinned upstream RLBench "
    "Git source commit f214b6ecc177babce12d3ad13dd92049c824c1d9 has not produced "
    "envs/uv/locks/rlbench.lock.txt in this workspace, and runtime smoke also requires "
    "an external CoppeliaSim/PyRep installation. See docs/rlbench_colosseum_integration.md."
)


BACKENDS: dict[str, dict[str, object]] = {
    "core": {
        "python": "3.11",
        "input": "core.in",
        "support_level": "uv",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_dummy_episode.py", "tests/test_control_interface.py", "tests/test_mapping.py", "tests/test_frame_transform.py"]],
    },
    "gymnasium": {
        "python": "3.11",
        "input": "gymnasium.in",
        "support_level": "uv",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_gymnasium_controller.py", "tests/test_dummy_episode.py"]],
    },
    "dmcontrol": {
        "python": "3.11",
        "input": "dmcontrol.in",
        "support_level": "uv",
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_dmcontrol_integration.py"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/dmcontrol_example.yaml"],
        ],
    },
    "robosuite": {
        "python": "3.10",
        "input": "robosuite.in",
        "support_level": "uv",
        "env": {"NUMBA_DISABLE_JIT": "1"},
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_robosuite_integration.py"]],
    },
    "robocasa": {
        "python": "3.11",
        "input": "robocasa.in",
        "support_level": "uv_source",
        "platform": "x86_64-manylinux_2_31",
        "torch_backend": "cpu",
        "excludes": ["robocasa-excludes.in"],
        "env": {"NUMBA_DISABLE_JIT": "1"},
        "sources": [
            {
                "name": "robosuite",
                "repo": "https://github.com/ARISE-Initiative/robosuite.git",
                "commit": "85abee228d1c43ab1939bce33028099945d453b4",
                "install_args": ["--no-deps", "-e"],
                "target_backend": "robocasa",
            },
            {
                "name": "robocasa",
                "repo": "https://github.com/robocasa/robocasa.git",
                "commit": "8f3c96ec8d1bfcd8126cad2bca887da98d30e997",
                "install_args": ["--no-deps", "-e"],
                "target_backend": "robocasa",
            },
        ],
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_robocasa_integration.py", "-rs"]],
    },
    "maniskill-cpu": {
        "python": "3.11",
        "input": "maniskill-cpu.in",
        "support_level": "uv",
        "torch_backend": "cpu",
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_maniskill_backend.py"]],
    },
    "maniskill-cu126": {
        "python": "3.11",
        "input": "maniskill-cu126.in",
        "support_level": "external_runtime",
        "torch_backend": "cu126",
        "smoke": [
            ["python", "-c", CUDA_SMOKE],
            ["python", "-m", "pytest", "-q", "tests/test_maniskill_backend.py"],
        ],
    },
    "metaworld": {
        "python": "3.11",
        "input": "metaworld.in",
        "support_level": "uv",
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_metaworld_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/metaworld_reach.yaml"],
        ],
    },
    "simpler-env": {
        "python": "3.10",
        "input": "simpler-env.in",
        "support_level": "uv_source",
        "sources": [
            {
                "name": "mani-skill2-real2sim",
                "repo": "https://github.com/simpler-env/ManiSkill2_real2sim.git",
                "commit": "ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682",
                "install_args": ["--no-deps", "-e"],
                "target_backend": "simpler-env",
            },
            {
                "name": "simpler-env",
                "repo": "https://github.com/simpler-env/SimplerEnv.git",
                "commit": "06accaca93535902d408da4855f21cece12bceb7",
                "install_args": ["--no-deps", "-e"],
                "target_backend": "simpler-env",
            },
        ],
        "smoke": [["python", "-m", "pytest", "-q", "tests/test_simpler_env_backend.py", "-rs"]],
    },
    "calvin": {
        "python": "3.10",
        "input": "calvin.in",
        "support_level": "uv_source",
        "sources": [
            {
                "name": "calvin-env",
                "repo": "https://github.com/mees/calvin_env.git",
                "commit": "797142c588c21e76717268b7b430958dbd13bf48",
                "install_args": ["--no-deps", "-e"],
                "target_backend": "calvin",
                "submodules": True,
            },
        ],
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_calvin_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/calvin_delta_eef.yaml"],
        ],
    },
    "rlbench": {
        "python": "3.10",
        "input": "rlbench.in",
        "blocked_reason": RLBENCH_UV_BLOCKED,
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_rlbench_colosseum_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/rlbench_reach_target_delta_eef.yaml"],
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
        for line in list_backends():
            print(line)
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


def backend_support_level(name: str) -> str:
    return str(backend_config(name).get("support_level", "uv"))


def backend_sources(name: str) -> list[dict[str, object]]:
    return list(backend_config(name).get("sources", []))  # type: ignore[arg-type]


def list_backends() -> list[str]:
    return [f"{name}\t{backend_support_level(name)}" for name in BACKENDS]


def require_unblocked_backend(name: str, command: str) -> None:
    cfg = backend_config(name)
    blocked_reason = cfg.get("blocked_reason")
    if not blocked_reason:
        return
    doc = cfg.get("doc")
    message = f"{name} uv {command} is blocked: {blocked_reason}"
    if doc:
        message = f"{message} See {doc}."
    raise SystemExit(message)


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
    if platform == PYTHON_PLATFORM and "platform" in cfg:
        platform = str(cfg["platform"])
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
    for exclude in cfg.get("excludes", []):  # type: ignore[union-attr]
        command.extend(["--excludes", str(UV_DIR / str(exclude))])
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
    install_backend_sources(name)


def install_backend_sources(name: str) -> None:
    for source in backend_sources(name):
        install_source(name, source)


def install_source(default_backend: str, source: dict[str, object]) -> None:
    validate_source_config(source)
    source_dir = source_path(source)
    ensure_git_checkout(source_dir, str(source["repo"]), str(source["commit"]))
    if source.get("submodules"):
        run(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)
    target_backend = str(source.get("target_backend", default_backend))
    install_args = [str(arg) for arg in source.get("install_args", ["--no-deps", "-e"])]  # type: ignore[union-attr]
    run(["uv", "pip", "install", "--python", str(venv_path(target_backend)), *install_args, str(source_dir)])


def validate_source_config(source: dict[str, object]) -> None:
    missing = [key for key in ("name", "repo", "commit", "install_args", "target_backend") if key not in source]
    if missing:
        raise SystemExit(f"source config is missing required keys: {', '.join(missing)}")
    validate_exact_commit(str(source["commit"]), str(source["name"]))


def validate_exact_commit(commit: str, source_name: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise SystemExit(f"{source_name} source commit must be a 40-character Git commit SHA: {commit}")


def source_path(source: dict[str, object]) -> Path:
    return VENV_DIR / "_sources" / str(source["name"])


def ensure_git_checkout(source_dir: Path, repo: str, commit: str) -> None:
    validate_exact_commit(commit, source_dir.name)
    if source_dir.exists():
        if not (source_dir / ".git").exists():
            raise SystemExit(f"source directory exists but is not a Git checkout: {source_dir}")
        run(["git", "remote", "set-url", "origin", repo], cwd=source_dir)
    else:
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir()
        run(["git", "init"], cwd=source_dir)
        run(["git", "remote", "add", "origin", repo], cwd=source_dir)
    run(["git", "fetch", "--depth", "1", "origin", commit], cwd=source_dir)
    run(["git", "checkout", "--detach", commit], cwd=source_dir)
    verify_git_checkout(source_dir, commit)


def verify_git_checkout(source_dir: Path, commit: str) -> None:
    actual = run_output(["git", "rev-parse", "HEAD"], cwd=source_dir)
    if actual != commit:
        raise SystemExit(f"source checkout {source_dir} expected {commit}, got {actual}")


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
    env.update(cfg.get("env", {}))  # type: ignore[arg-type]
    if name == "robocasa":
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
    return env


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd or ROOT), env=env, check=True)


def run_output(command: list[str], *, cwd: Path | None = None) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(cwd or ROOT), check=True, capture_output=True, text=True)
    return result.stdout.strip()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
