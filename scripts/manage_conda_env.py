#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venvs"
SOURCE_DIR = VENV_DIR / "_sources"
LIBERO_CONFIG_DIR = VENV_DIR / "libero_config"
CONDA_EXE = Path(os.environ.get("CONDA_EXE", "/home/caozhe/miniconda3/bin/conda"))
COPPELIASIM_URL = "https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
COPPELIASIM_ROOT = Path(os.environ.get("COPPELIASIM_ROOT", "/home/caozhe/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"))


BACKENDS: dict[str, dict[str, object]] = {
    "rlbench": {
        "env_name": "mllm-teleop-rlbench",
        "env_file": "envs/environment-rlbench.yml",
        "require_flag": "MLLM_TELEOP_REQUIRE_RLBENCH",
        "root_install_args": ["-e"],
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
    "colosseum": {
        "env_name": "mllm-teleop-colosseum",
        "env_file": "envs/environment-colosseum.yml",
        "require_flag": "MLLM_TELEOP_REQUIRE_COLOSSEUM",
        "root_install_args": ["--no-deps", "-e"],
        "sources": [
            {
                "name": "colosseum-pyrep",
                "repo": "https://github.com/stepjam/PyRep.git",
                "commit": "076ca15c57f2495a4194da03565891ab1aaa317e",
                "install_args": ["--no-build-isolation", "--no-deps", "-e"],
            },
            {
                "name": "colosseum-rlbench",
                "repo": "https://github.com/stepjam/RLBench.git",
                "commit": "7c3f425f4a0b6b5ce001ba7246354eb3c70555be",
                "install_args": ["--no-deps", "-e"],
            },
            {
                "name": "colosseum",
                "repo": "https://github.com/robot-colosseum/robot-colosseum.git",
                "commit": "098a910ae1d490524adb49b8d50735b00a1a62de",
                "install_args": ["--no-deps", "-e"],
            },
        ],
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_colosseum_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/colosseum_close_box_manip_obj_color_smoke.yaml"],
        ],
    },
    "libero": {
        "env_name": "mllm-teleop-libero",
        "env_file": "envs/environment-libero.yml",
        "require_flag": "MLLM_TELEOP_REQUIRE_LIBERO",
        "root_install_args": ["--no-deps", "-e"],
        "sources": [
            {
                "name": "libero",
                "repo": "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
                "commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
                "install_args": ["--no-deps", "-e"],
            },
        ],
        "smoke": [
            ["python", "-m", "pytest", "-q", "tests/test_libero_integration.py", "-rs"],
            ["python", "scripts/run_benchmark.py", "--config", "configs/libero_spatial_task0.yaml"],
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
        report_runtime(ns.backend)
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
    if name == "libero":
        checkout_backend_sources(name)
        install_libero_runtime_dependencies(name)
    install_backend_sources(name)
    root_install_args = [str(arg) for arg in cfg.get("root_install_args", ["-e"])]  # type: ignore[union-attr]
    run_conda(name, ["python", "-m", "pip", "install", *root_install_args, str(ROOT)])
    if name == "libero":
        install_libero_source_path(name)
        write_libero_config(name)


def conda_env_exists(name: str) -> bool:
    output = run_output([str(CONDA_EXE), "env", "list"])
    return any(line.split() and line.split()[0] == name for line in output.splitlines())


def checkout_backend_sources(name: str) -> None:
    for source in backend_config(name).get("sources", []):  # type: ignore[union-attr]
        ensure_git_checkout(source_path(source), str(source["repo"]), str(source["commit"]))


def install_backend_sources(name: str) -> None:
    for source in backend_config(name).get("sources", []):  # type: ignore[union-attr]
        install_source(name, source)


def install_libero_runtime_dependencies(name: str) -> None:
    if name != "libero":
        return
    runtime_requirement_batches = [
        ["numpy==1.22.4", "PyYAML>=6.0", "imageio>=2.0", "pytest>=8.0,<8.4", "pygame>=2.5"],
        ["future==0.18.2", "cloudpickle==2.1.0", "gym==0.25.2", "easydict==1.9", "hydra-core==1.2.0"],
        ["opencv-python==4.6.0.66", "bddl==1.0.1", "matplotlib==3.5.3"],
        ["numpy==1.22.4", "scipy==1.10.1", "numba==0.57.1", "mujoco==2.3.7", "termcolor>=2.0", "robosuite==1.4.0"],
    ]
    for requirements in runtime_requirement_batches:
        run_conda(name, ["python", "-m", "pip", "install", "--prefer-binary", *requirements])
    run_conda(name, ["python", "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu", "torch==1.11.0+cpu"])
    setup_robosuite_macros(name)


def setup_robosuite_macros(name: str) -> None:
    if name != "libero":
        return
    script = (
        "from pathlib import Path; "
        "import robosuite; "
        "base = Path(robosuite.__path__[0]); "
        "source = base / 'macros.py'; "
        "target = base / 'macros_private.py'; "
        "assert source.exists(), source; "
        "target.write_text(source.read_text(encoding='utf-8'), encoding='utf-8')"
    )
    run_conda(name, ["python", "-c", script])


def install_source(name: str, source: dict[str, object]) -> None:
    source_dir = source_path(source)
    ensure_git_checkout(source_dir, str(source["repo"]), str(source["commit"]))
    install_args = [str(arg) for arg in source.get("install_args", ["-e"])]  # type: ignore[union-attr]
    run_conda(name, ["python", "-m", "pip", "install", *install_args, str(source_dir)])


def install_libero_source_path(name: str) -> None:
    libero_package_root = SOURCE_DIR / "libero"
    script = (
        "from pathlib import Path; "
        "import site; "
        f"source = Path({str(libero_package_root)!r}); "
        "assert (source / 'libero' / 'libero' / '__init__.py').exists(), source; "
        "site_packages = Path(site.getsitepackages()[0]); "
        "(site_packages / 'libero_source.pth').write_text(str(source) + '\\n', encoding='utf-8')"
    )
    run_conda(name, ["python", "-c", script])


def source_path(source: dict[str, object]) -> Path:
    return SOURCE_DIR / str(source["name"])


def ensure_git_checkout(source_dir: Path, repo: str, commit: str) -> None:
    validate_exact_commit(commit, source_dir.name)
    if source_dir.exists():
        if not (source_dir / ".git").exists():
            raise SystemExit(f"source directory exists but is not a Git checkout: {source_dir}")
        try:
            actual = run_output(["git", "rev-parse", "HEAD"], cwd=source_dir)
        except subprocess.CalledProcessError:
            actual = ""
        if actual == commit:
            verify_git_checkout(source_dir, commit)
            return
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
    status = run_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source_dir,
    )
    if status:
        raise SystemExit(
            f"source checkout {source_dir} has tracked modifications; "
            "refusing to install a dirty pinned source"
        )


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


def write_libero_config(name: str) -> None:
    benchmark_root = SOURCE_DIR / "libero" / "libero" / "libero"
    script = (
        "from pathlib import Path; "
        "import yaml; "
        f"root = Path({str(benchmark_root)!r}); "
        "assert (root / '__init__.py').exists(), root; "
        f"cfg_dir = Path({str(LIBERO_CONFIG_DIR)!r}); "
        "cfg_dir.mkdir(parents=True, exist_ok=True); "
        "yaml.safe_dump({"
        "'benchmark_root': str(root), "
        "'bddl_files': str(root / 'bddl_files'), "
        "'init_states': str(root / 'init_files'), "
        "'datasets': str(root.parent / 'datasets'), "
        "'assets': str(root / 'assets')"
        "}, (cfg_dir / 'config.yaml').open('w', encoding='utf-8'), sort_keys=True)"
    )
    run_conda(name, ["python", "-c", script])


def smoke_backend(name: str) -> None:
    env = backend_env(name)
    env[str(backend_config(name)["require_flag"])] = "1"
    for command in backend_config(name)["smoke"]:  # type: ignore[index]
        run_backend(name, list(command), env=env)


def run_backend(name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    if command[0] == "python":
        command = ["python", *command[1:]]
    run_conda(name, command, env=env)


def run_conda(name: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    run([str(CONDA_EXE), "run", "-n", env_name(name), *command], env=env or backend_env(name))


def backend_env(name: str) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(ROOT / "src")]
    if name == "libero":
        libero_source_root = SOURCE_DIR / "libero"
        libero_benchmark_root = libero_source_root / "libero" / "libero"
        python_paths.append(str(libero_source_root))
        env["MLLM_TELEOP_LIBERO_BENCHMARK_ROOT"] = str(libero_benchmark_root)
        env["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["NUMBA_DISABLE_JIT"] = "1"
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
    if name in {"rlbench", "colosseum"}:
        env["COPPELIASIM_ROOT"] = str(COPPELIASIM_ROOT)
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIASIM_ROOT)
        env["LD_LIBRARY_PATH"] = f"{COPPELIASIM_ROOT}:{env.get('LD_LIBRARY_PATH', '')}"
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if "DISPLAY" not in env and Path("/tmp/.X11-unix/X1").exists():
        env["DISPLAY"] = ":1"
    return env


def report_runtime(name: str | None = None) -> None:
    require_conda()
    names = [name] if name else list(BACKENDS)
    for backend in names:
        env = backend_env(backend)
        print(f"[{backend}]")
        print(f"conda={CONDA_EXE}")
        print(f"env_name={env_name(backend)}")
        print(f"display={env.get('DISPLAY', '')}")
        if backend in {"rlbench", "colosseum"}:
            print(f"coppeliasim_root={COPPELIASIM_ROOT}")
            print(f"coppeliasim_exists={(COPPELIASIM_ROOT / 'coppeliaSim.sh').exists()}")
            print(f"ld_library_path={env.get('LD_LIBRARY_PATH', '')}")
        if backend == "libero":
            print(f"libero_config={LIBERO_CONFIG_DIR}")
            print(f"mujoco_gl={env.get('MUJOCO_GL', '')}")
            print(f"pyopengl_platform={env.get('PYOPENGL_PLATFORM', '')}")
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
