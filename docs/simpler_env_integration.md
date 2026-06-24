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

- `simpler_env` from `https://github.com/simpler-env/SimplerEnv`
- `mani_skill2_real2sim` from `https://github.com/simpler-env/ManiSkill2_real2sim`

It intentionally excludes RT-1, Octo, TensorFlow/JAX policy inference packages, and SimplerEnv's full-install dependencies. Core tests and non-SimplerEnv environments continue to work when `simpler_env` is not installed.

SimplerEnv's upstream README lists CUDA and NVIDIA GPU prerequisites for practical SAPIEN rendering. Hosts without a working GPU driver, compatible CUDA/SAPIEN wheels, or required upstream assets may skip or fail the runtime smoke while config/controller tests still pass.

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
- Human direct play is not wired as a separate `direct_simpler_env` backend yet.
- The uv backend has no policy inference stack; it is for environment reset/step/render integration only.
