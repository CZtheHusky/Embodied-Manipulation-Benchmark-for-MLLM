# SimplerEnv Integration

SimplerEnv is integrated as an optional real-to-sim/VLA evaluation backend. The benchmark does not vendor or modify upstream SimplerEnv or ManiSkill2 real-to-sim source code.

## Components

- Adapter: `sim/simpler_env/adapter.py`
- Observation provider: `sim/simpler_env/observation.py`
- Controller: `control/simpler_env.py`
- Task wrapper: `tasks/simpler_env_task.py`
- Example config: `configs/simpler_env_google_robot_pick_coke_can.yaml`
- uv backend: `simpler-env`

## Dependency

Use the dedicated uv backend:

```bash
python3 scripts/manage_uv_env.py create simpler-env
python3 scripts/manage_uv_env.py smoke simpler-env
```

The backend installs only the SimplerEnv environment runtime:

- PyPI runtime dependencies from `envs/uv/simpler-env.in`, locked in `envs/uv/locks/simpler-env.lock.txt`
- `mani_skill2_real2sim` from `https://github.com/simpler-env/ManiSkill2_real2sim` at commit `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`
- `simpler_env` from `https://github.com/simpler-env/SimplerEnv` at commit `06accaca93535902d408da4855f21cece12bceb7`

It intentionally excludes RT-1, Octo, TensorFlow/JAX policy inference packages, and SimplerEnv's full-install dependencies. Core tests and non-SimplerEnv environments continue to work when `simpler_env` is not installed.

The uv lock intentionally does not contain Git URL requirements for the two upstream source packages. Direct uv resolution against those Git repositories stalled while updating `ManiSkill2_real2sim`; `create simpler-env` instead fetches the pinned commits into `.venvs/_sources/` and installs them editable with `--no-deps` after syncing the lock.

`sapien==2.2.2` imports `pkg_resources`, so the backend pins `setuptools<81`. Removing that pin makes `simpler_env` fail during import.

SimplerEnv's upstream README lists CUDA and NVIDIA GPU prerequisites for practical SAPIEN rendering. Hosts without a working GPU driver, compatible CUDA/SAPIEN wheels, or required upstream assets should fail the runtime smoke rather than reporting runtime verification. Config/controller tests still pass when `simpler_env` is not installed.

Verified on this branch on 2026-06-24:

```bash
python3 scripts/manage_uv_env.py lock simpler-env
python3 scripts/manage_uv_env.py create simpler-env
python3 scripts/manage_uv_env.py smoke simpler-env
```

The smoke run passed `6 passed` and exercised `simpler_env` import, reset, RGB observation/render extraction, one zero action step, and close.

Manual play uses the shared Pygame worker runtime:

```bash
python3 scripts/manage_uv_env.py run simpler-env -- python scripts/play_benchmark.py --config configs/simpler_env_google_robot_pick_coke_can.yaml
```

On this station the SAPIEN runtime smoke requires access to the desktop X11 session, for example `DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority`. Running the same command inside a sandbox that cannot open the X display fails before benchmark code runs.

## Initial Task

The initial config targets upstream task name `google_robot_pick_coke_can`, which maps upstream to `GraspSingleOpenedCokeCanInScene-v0`. Upstream currently documents this task in the minimal example:

```python
import simpler_env
env = simpler_env.make("google_robot_pick_coke_can")
```

## Control

The benchmark control path remains simulator-independent:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> SimplerEnvDeltaEEFController -> env.step()
```

Joystick permutation, sign changes, and coordinate-frame rotation are handled by this repository before native SimplerEnv actions are produced. The controller always emits a 7D action in this order:

```text
dx, dy, dz, droll, dpitch, dyaw, gripper
```

The rotation slots are passed into SimplerEnv's delta rotation action fields. Position and rotation scales are configured in YAML, and the adapter clips the final action against the native environment action space when available.

The gripper is the final virtual joystick. The default mapping is `0.0 -> -1.0` and `1.0 -> 1.0`, configurable through controller params.

## Observation

The observation provider first tries to extract RGB frames from the native observation dict, then falls back to SimplerEnv's official `get_image_from_maniskill2_obs_dict(env, obs)` helper, and finally to `env.render()`.

The adapter stores the latest native observation, reward, info, `success`, `truncated`, and language instruction when `env.get_language_instruction()` is available. `simpler_env_generic` uses `task.params.success_key`, defaulting to `success`, for success and metrics.

## Current Limitations

- Initial support is single-environment and Box-action focused.
- Runtime camera mutation is out of scope; the initial config uses the default SimplerEnv rendered image.
- The uv backend has no policy inference stack; it is for environment reset/step/render integration only.
