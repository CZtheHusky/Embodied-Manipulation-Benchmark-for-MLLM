# SimplerEnv Integration

SimplerEnv is integrated as an optional real-to-sim/VLA evaluation backend. The benchmark does not vendor or modify upstream SimplerEnv or ManiSkill2 real-to-sim source code.

## Components

- Adapter: `sim/simpler_env/adapter.py`
- Observation provider: `sim/simpler_env/observation.py`
- Controller: `control/simpler_env.py`
- Task wrapper: `tasks/simpler_env_task.py`
- Example config: `configs/simpler_env_google_robot_pick_coke_can.yaml`
- MLLM baseline: `configs/simpler_env_google_robot_pick_coke_can_mllm_reflection.yaml`
- MLLM perturbation: `configs/simpler_env_google_robot_pick_coke_can_mllm_reflection_perturbed.yaml`
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

For the Google Robot task, upstream defines the first three native action
values as delta XYZ, the next three as an axis-angle rotation vector, and the
last value as the gripper command. The active upstream controller uses an
`ee_align` frame: the control origin is at the end effector and the axes are
aligned to the robot base. In the benchmark's public joystick schema,
`joystick_6=0` means open and `joystick_6=1` means closed; the controller maps
those values to upstream `-1` and `1`.

## Observation

The observation provider first tries to extract RGB frames from the native observation dict, then falls back to SimplerEnv's official `get_image_from_maniskill2_obs_dict(env, obs)` helper, and finally to `env.render()`.

The adapter stores the latest native observation, reward, info, `success`, `truncated`, and language instruction when `env.get_language_instruction()` is available. `simpler_env_generic` uses `task.params.success_key`, defaulting to `success`, for success and metrics.

Some upstream task diagnostics contain `sapien.Pose` values. The adapter
serializes those evaluator-only values as `position` and
`quaternion_wxyz` before trajectory logging; they remain excluded from model
requests.

The MLLM configs explicitly select only the public RGB views
`image/base_camera` and `image/overhead_camera`. Native segmentation tensors
remain in the simulator observation but are not selected by the MLLM
projection. Model requests contain the task text, public control contract,
selected RGB, and the model's own action/reflection history. They do not
contain simulator state, reward, success, distance, object or end-effector
pose, native simulator action, seed, or sampled mapping/frame-transform
parameters.

## MLLM Evaluation

Both MLLM configurations use the shared `MLLMPolicy` and `SimAdapter` path,
OpenAI-compatible Chat Completions, `action_with_reflection`,
`reasoning_effort=low`, three historical RGB steps, and complete public action
history. Video logging is disabled by default.

The baseline exposes the fixed joystick mapping and uses the identity
benchmark frame transform. The perturbed configuration uses a hidden
`permuted_sign_flipped` motion mapping and hidden `random_rotation_3d`
transform. The gripper remains the final, unpermuted, non-sign-flipped
joystick in both modes. The sampled permutation, signs, and rotation remain
fixed within one episode and are never included in model requests.

Run the environment integration tests with the pinned runtime:

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority \
PYTHONPATH=src .venvs/simpler-env/bin/python \
  -m pytest tests/test_simpler_env_backend.py \
  tests/test_simpler_env_integration.py -q
```

The tests perform real reset/step/render operations and run both MLLM
configurations for four steps with an in-process fake model client. They check
native action shape and bounds, RGB variance, image history, trajectory and
run manifests, and privileged-information isolation. They do not access a
model API.

## Current Limitations

- Initial support is single-environment and Box-action focused.
- Runtime camera mutation is out of scope; the MLLM configs consume the two
  RGB cameras produced by the pinned Google Robot task.
- The uv backend has no policy inference stack; it is for environment reset/step/render integration only.
