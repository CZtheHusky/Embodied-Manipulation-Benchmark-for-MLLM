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

Dependency lock status on 2026-06-24: `python3 scripts/manage_uv_env.py lock libero` started `uv pip compile` with Python 3.10 and `--torch-backend cu113`, updated the pinned LIBERO Git dependency, then remained in dependency resolution with no further output for multiple 30-second polling windows. The command was interrupted to keep this integration minimal. No `envs/uv/locks/libero.lock.txt` is checked in yet.

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

Use uv isolation:

```bash
python3 scripts/manage_uv_env.py create libero
python3 scripts/manage_uv_env.py smoke libero
```

`envs/uv/libero.in` pins upstream LIBERO to commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` and follows the old official stack as closely as this project can:

- Torch `1.11.0`
- torchvision `0.12.0`
- torchaudio `0.11.0`
- CUDA backend `cu113`
- robosuite `1.4.0`
- gym `0.25.2`
- bddl `1.0.1`

Known risk: official LIBERO asks for Python 3.8.13, but this repository's package metadata requires Python >=3.10. The uv backend therefore uses Python 3.10. If resolution or runtime fails, use the failure as a blocker for this backend rather than weakening the global project Python requirement.

Current blocker: the first uv lock attempt did not complete in this environment before being interrupted. Treat `create libero` and real runtime smoke as blocked until `envs/uv/locks/libero.lock.txt` is generated or an official conda fallback is used.

## Smoke Test

The runtime smoke in `tests/test_libero_integration.py` skips when LIBERO, robosuite, BDDL files, init states, assets, or headless rendering are unavailable. When the stack is installed, it verifies:

- config and controller build
- reset
- `set_init_state`
- one 7D action step
- image extraction or render
- close
