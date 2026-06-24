#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venvs"
SOURCE_DIR = VENV_DIR / "_sources"
CONDA_EXE = Path(os.environ.get("CONDA_EXE", "/home/caozhe/miniconda3/bin/conda"))
COPPELIASIM_URL = "https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
COPPELIASIM_ROOT = Path(
    os.environ.get("COPPELIASIM_ROOT", "/home/caozhe/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04")
)


BACKENDS: dict[str, dict[str, object]] = {
    "rlbench": {
        "env_name": "mllm-teleop-rlbench",
        "env_file": "envs/environment-rlbench.yml",
        "sources": [
            {
                "name": "pyrep",
                "repo": "https://github.com/stepjam/PyRep.git",
                "commit": "076ca15c57f2495a4194da03565891ab1aaa317e",
                "install_args": ["--no-build-isolation", "-e"],
            },
            {
                "name": "rlbench",
                "repo": "https://github.com/stepjam/RLBench.git",
                "commit": "f214b6ecc177babce12d3ad13dd92049c824c1d9",
                "install_args": ["-e"],
            },
        ],
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_rlbench_colosseum_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/rlbench_reach_target_delta_eef_smoke.yaml"],
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Miniconda fallback environments for simulator backends.")
    parser.add_argument("command", choices=("create", "run", "smoke", "list", "install-coppeliasim", "report"))
    parser.add_argument("backend", nargs="?", choices=tuple(BACKENDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    if ns.command == "list":
        for name, cfg in BACKENDS.items():
            print(f"{name}\t{cfg['env_name']}")
        return
    if ns.command == "install-coppeliasim":
        install_coppeliasim()
        return
    if ns.command == "report":
        report_runtime()
        return
    if ns.backend is None:
        parser.error("backend is required")

    if ns.command == "create":
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


def env_name(name: str) -> str:
    return str(backend_config(name)["env_name"])


def create_backend(name: str) -> None:
    cfg = backend_config(name)
    require_conda()
    env_file = str(ROOT / str(cfg["env_file"]))
    if conda_env_exists(env_name(name)):
        run([str(CONDA_EXE), "env", "update", "-n", env_name(name), "-f", env_file, "--prune"])
    else:
        run([str(CONDA_EXE), "env", "create", "-f", env_file])
    install_backend_sources(name)
    run_conda(name, ["python", "-m", "pip", "install", "-e", str(ROOT)])


def conda_env_exists(name: str) -> bool:
    output = run_output([str(CONDA_EXE), "env", "list"])
    return any(line.split() and line.split()[0] == name for line in output.splitlines())


def install_backend_sources(name: str) -> None:
    for source in backend_config(name).get("sources", []):  # type: ignore[union-attr]
        install_source(name, source)


def install_source(name: str, source: dict[str, object]) -> None:
    source_dir = source_path(source)
    ensure_git_checkout(source_dir, str(source["repo"]), str(source["commit"]))
    install_args = [str(arg) for arg in source.get("install_args", ["-e"])]  # type: ignore[union-attr]
    run_conda(name, ["python", "-m", "pip", "install", *install_args, str(source_dir)])


def source_path(source: dict[str, object]) -> Path:
    return SOURCE_DIR / str(source["name"])


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
    actual = run_output(["git", "rev-parse", "HEAD"], cwd=source_dir)
    if actual != commit:
        raise SystemExit(f"source checkout {source_dir} expected {commit}, got {actual}")


def validate_exact_commit(commit: str, source_name: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise SystemExit(f"{source_name} source commit must be a 40-character Git commit SHA: {commit}")


def install_coppeliasim() -> None:
    archive = Path("/tmp") / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
    COPPELIASIM_ROOT.mkdir(parents=True, exist_ok=True)
    if not (COPPELIASIM_ROOT / "coppeliaSim.sh").exists():
        run(["curl", "-L", COPPELIASIM_URL, "-o", str(archive)])
        run(["tar", "-xf", str(archive), "-C", str(COPPELIASIM_ROOT), "--strip-components", "1"])
    print(COPPELIASIM_ROOT)


def smoke_backend(name: str) -> None:
    env = backend_env()
    env["MLLM_TELEOP_REQUIRE_RLBENCH"] = "1"
    for command in backend_config(name)["smoke"]:  # type: ignore[index]
        run_backend(name, list(command), env=env)


def run_backend(name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    if command[0] == "python":
        command = ["python", *command[1:]]
    run_conda(name, command, env=env)


def run_conda(name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    run([str(CONDA_EXE), "run", "-n", env_name(name), *command], env=env or backend_env())


def backend_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["COPPELIASIM_ROOT"] = str(COPPELIASIM_ROOT)
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIASIM_ROOT)
    env["LD_LIBRARY_PATH"] = f"{COPPELIASIM_ROOT}:{env.get('LD_LIBRARY_PATH', '')}"
    if "DISPLAY" not in env and Path("/tmp/.X11-unix/X1").exists():
        env["DISPLAY"] = ":1"
    return env


def report_runtime() -> None:
    env = backend_env()
    print(f"conda={CONDA_EXE}")
    print(f"coppeliasim_root={COPPELIASIM_ROOT}")
    print(f"coppeliasim_exists={(COPPELIASIM_ROOT / 'coppeliaSim.sh').exists()}")
    print(f"display={env.get('DISPLAY', '')}")
    print(f"ld_library_path={env.get('LD_LIBRARY_PATH', '')}")
    run(["bash", "-lc", "nvidia-smi || true"])


def require_conda() -> None:
    if not CONDA_EXE.exists():
        raise SystemExit(f"Miniconda executable does not exist: {CONDA_EXE}")


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
