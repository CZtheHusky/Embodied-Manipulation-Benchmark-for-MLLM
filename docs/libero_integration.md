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

Dependency status on 2026-06-24: LIBERO is runtime verified through the Miniconda fallback in this branch. The uv setup remains blocked by the old upstream dependency stack and is not the supported route for this backend.

Verified commands:

```bash
python3 scripts/manage_conda_env.py create libero
python3 scripts/manage_conda_env.py smoke libero
```

The smoke result was:

- `tests/test_libero_integration.py`: `5 passed`
- `configs/libero_spatial_task0.yaml`: one random-agent benchmark episode, `average_steps=200.0`, `success_rate=0.0`
- `scripts/play_benchmark.py --config configs/libero_spatial_task0.yaml --set ui.max_steps=0`: shared Pygame worker runtime starts LIBERO, extracts the `agentview` frame, and exits cleanly.

The benchmark success rate is expected to be zero for the random policy. The verification claim is reset, init-state application, one environment step, image/render access, close, and benchmark-runner completion.

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

That official Python 3.8.13 environment is useful for validating upstream LIBERO itself, but it is not the runtime for this repository because `pyproject.toml` requires Python >=3.10. Do not weaken the repository-wide Python requirement to fit LIBERO.

This repository uses a separate Miniconda compatibility environment that keeps Python 3.10 and installs only the runtime subset needed for `OffScreenRenderEnv`, benchmark metadata, and the benchmark runner:

```bash
python3 scripts/manage_conda_env.py create libero
python3 scripts/manage_conda_env.py smoke libero
```

Manual play uses the same conda backend:

```bash
python3 scripts/manage_conda_env.py run libero -- python scripts/play_benchmark.py --config configs/libero_spatial_task0.yaml
```

The script pins upstream LIBERO to commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` and writes a `.pth` file because that repository's root `setup.py` installs metadata but does not expose the nested `libero.libero` package correctly in this Python 3.10 editable install.

The verified compatibility stack is:

- Torch `1.11.0+cpu`
- robosuite `1.4.0`
- MuJoCo `2.3.7`
- NumPy `1.22.4`
- SciPy `1.10.1`
- numba `0.57.1`
- gym `0.25.2`
- bddl `1.0.1`
- matplotlib `3.5.3`

This is deliberately not the full LIBERO training stack. `wandb`, `transformers`, `robomimic`, `torchvision`, and `torchaudio` are not needed for the runtime adapter smoke and should not be pulled into this environment unless a policy-training/evaluation workflow is added and verified separately.

Proxy note: pip downloads used `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897`. The conda helper removes `ALL_PROXY/all_proxy` so pip does not accidentally take a SOCKS path through Clash.

## Smoke Test

The runtime smoke in `tests/test_libero_integration.py` skips when LIBERO is not required and LIBERO, robosuite, BDDL files, init states, assets, or headless rendering are unavailable. Set `MLLM_TELEOP_REQUIRE_LIBERO=1` in a claimed LIBERO runtime environment so missing dependencies or runtime failures fail loudly. When the stack is installed, it verifies:

- config and controller build
- reset
- `set_init_state`
- one 7D action step
- image extraction or render
- close
