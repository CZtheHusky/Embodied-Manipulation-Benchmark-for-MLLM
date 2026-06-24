# LIBERO Integration

LIBERO is added as an optional language-conditioned manipulation backend. The goal is a minimal reproducible adapter for evaluating closed-loop visual feedback, not a claim that standard LIBERO scores are robust language-understanding measurements.

## Status

Initial config:

- `configs/libero_spatial_task0.yaml`
- `simulator.type: libero`
- `task.type: libero_generic`
- `controller.type: libero_delta_eef`
- Suite/task/init state: `libero_spatial`, task `0`, init-state `0`

The official API checked for this adapter is:

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=task_suite.get_task_bddl_file_path(0))
env.reset()
env.set_init_state(task_suite.get_task_init_states(0)[0])
obs, reward, done, info = env.step([0.0] * 7)
```

This matches the upstream README and current `env_wrapper.py`: `OffScreenRenderEnv` wraps robosuite and exposes `reset`, `set_init_state`, `step`, `check_success`, `sim`, and `close`.

Dependency status on 2026-06-24: LIBERO uv setup is blocked and not runtime verified. `python3 scripts/manage_uv_env.py lock libero` starts `uv pip compile` with Python 3.10 and `--torch-backend cu113`, updates the pinned LIBERO Git dependency, then remains in the upstream Git source fetch/resolution path without completing. Follow-up probes showed the direct pinned dependency stack without `libero @ git+...` resolves 100 packages in about 15 seconds, while Git-only uv resolution, a GitHub archive URL, and direct Git source checkout did not complete cleanly in this environment. No `envs/uv/locks/libero.lock.txt` is checked in.

`scripts/manage_uv_env.py` intentionally fails fast for `lock`, `create`, `run`, and `smoke` on the `libero` backend until a lock and runtime smoke have both been verified.

## Architecture

The integration follows the project boundaries:

- `LiberoSimAdapter`: owns LIBERO/robosuite environment creation, reset, set-init-state, step, render, state extraction, and close.
- `LiberoDeltaEEFController`: owns joystick mapping, sign, scale, frame transform, and final 7D action creation.
- `LiberoObservationProvider`: extracts `*_image` RGB observations before falling back to `sim.render`.
- `LiberoTask`: provides instruction, success, metrics, and benchmark-validity metadata.

Heavy imports are lazy. Importing core registry, config loading, controllers, and tests does not import `libero`, `robosuite`, or `torch`.

## Control

The action path is:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> LiberoDeltaEEFController -> LiberoAction -> OffScreenRenderEnv.step()
```

The native action order is fixed:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

Default scales mirror robosuite OSC-style small deltas: `0.05` for translation and `0.1` for rotation. The gripper defaults to `0.0 -> -1.0` and `1.0 -> 1.0`.

## Benchmark Validity

Every reset records:

- language instruction
- suite name
- task id
- upstream task name
- init-state id
- `no_language_baseline`
- `random_baseline`

These fields are included in observation metadata, simulator state, and task metrics under `benchmark_validity`.

The default config uses the random agent and marks `random_baseline: true`. For a shortcut baseline that removes language, run with:

```bash
python3 scripts/run_benchmark.py --config configs/libero_spatial_task0.yaml --set task.params.no_language_baseline=true
```

Do not report LIBERO success without also reporting these metadata fields and the baseline setting. Recent robustness work has shown that standard LIBERO evaluation can reward memorized action/layout shortcuts, so LIBERO should be treated as a candidate benchmark until no-language, random, and perturbation baselines are compared.

## Dependencies

The uv backend is blocked:

```bash
python3 scripts/manage_uv_env.py create libero
python3 scripts/manage_uv_env.py smoke libero
```

Both commands fail fast by design. `envs/uv/libero.in` records the attempted Python 3.10 compatibility lock input and pins upstream LIBERO to commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, but this path has not produced a lock or runtime smoke.

The upstream official conda reproduction path is:

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

That official Python 3.8.13 environment is useful for validating upstream LIBERO itself, but it is not a verified runtime for this repository because `pyproject.toml` requires Python >=3.10. Do not weaken the repository-wide Python requirement to fit LIBERO.

For this repository, the only acceptable fallback is a separate conda compatibility environment that keeps Python 3.10 and then proves the adapter with required runtime smoke:

```bash
conda create -n mllm-teleop-libero python=3.10
conda activate mllm-teleop-libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
cd /home/caozhe/workspace/embodied-sim-libero
pip install -e .
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_DISABLE_JIT=1
MLLM_TELEOP_REQUIRE_LIBERO=1 python -m pytest -q tests/test_libero_integration.py -rs
python scripts/run_benchmark.py --config configs/libero_spatial_task0.yaml
```

Until those final two commands pass, report LIBERO as adapter-implemented but runtime-blocked.

The compatibility stack follows the old official dependencies as closely as this project can:

- Torch `1.11.0`
- torchvision `0.12.0`
- torchaudio `0.11.0`
- CUDA backend `cu113`
- robosuite `1.4.0`
- gym `0.25.2`
- bddl `1.0.1`

## Smoke Test

The runtime smoke in `tests/test_libero_integration.py` skips when LIBERO is not required and LIBERO, robosuite, BDDL files, init states, assets, or headless rendering are unavailable. Set `MLLM_TELEOP_REQUIRE_LIBERO=1` in a claimed LIBERO runtime environment so missing dependencies or runtime failures fail loudly. When the stack is installed, it verifies:

- config and controller build
- reset
- `set_init_state`
- one 7D action step
- image extraction or render
- close
