# RoboCasa Integration

RoboCasa / RoboCasa365 is the semantic household manipulation candidate backend. The minimal integration uses the official Gym wrapper style:

```python
gym.make("robocasa/PickPlaceCounterToCabinet", split="pretrain", seed=seed)
```

The first benchmark config is `configs/robocasa_pick_place_counter_to_cabinet.yaml`.

## Backend Boundary

- `simulator.type: robocasa` builds `RoboCasaSimAdapter`.
- `task.type: robocasa_generic` builds `RoboCasaGenericTask`.
- `controller.type: robocasa_osc` builds `RoboCasaOSCController`.
- Heavy imports stay inside `RoboCasaSimAdapter.reset`; core imports and config building do not import `robocasa`.

RoboCasa's Gym wrapper action space is a `Dict`, not the robosuite 7D OSC array. The controller emits:

- `action.end_effector_position`: 3D delta position.
- `action.end_effector_rotation`: 3D delta rotation.
- `action.gripper_close`: 1D close command in `[0, 1]`.
- `action.base_motion`: 4D zero vector for this minimal integration.
- `action.control_mode`: 1D mode value, default `1.0`.

The base-motion channels are intentionally fixed at zero for the first pass. Add mobile-base joystick dimensions only after validating the upstream controller semantics for the selected robot.

## MLLM Evaluation

The PickPlaceCounterToCabinet task has two switchable Chat Completions configs:

- `configs/robocasa_pick_place_counter_to_cabinet_mllm_chat_completions.yaml`: `fixed + identity`.
- `configs/robocasa_pick_place_counter_to_cabinet_mllm_chat_completions_perturbed.yaml`: hidden `permuted_sign_flipped + random_rotation_3d`.

Both use `chat_completions` with optional Reflection, reasoning effort `low`, three historical image
steps, all action history, and prompt contract
`mllm_joystick_chat_completions_v7`. After reset, both conditions project the upstream
public `annotation.human.task_description` string into the prompt. This names the
episode's target object and avoids the ambiguous static fallback when distractor
objects are present. A missing or empty upstream instruction is a runtime error.

The public MLLM joystick has six continuous `[-30, 30]` motion channels followed
by one binary gripper channel:

- PandaOmron OSC controls `robot0_right_center` in the robot-base frame. This
  frame moves with the mobile base; it is not a fixed world frame.
- `dx`, `dy`, and `dz` are delta translations in that moving base frame.
- `droll`, `dpitch`, and `dyaw` form an axis-angle rotation vector in the same
  moving base frame.
- Motion commands use the official normalized `[-1, 1]` OSC input. Upstream
  maps native magnitude `1` to approximately `0.05 m` translation and
  `0.5 rad` rotation. The benchmark therefore uses normalized scales of `0.2`
  and `0.0628318530717959`, producing physical magnitude-1 steps of `0.01 m`
  and `0.0314159265358979 rad` (`1.8 degrees`) without modifying the upstream
  controller gains.
- The baseline applies no backend sign compensation. Positive and negative
  commands are passed with their stated base-frame signs; the perturbed
  condition's explicit hidden sign flips are benchmark perturbations.
- The virtual gripper is binary: `0` means open and `1` means close. The pinned RoboCasa wrapper internally maps values below `0.5` to native `-1` and values at or above `0.5` to native `1`; there is no distinct `0.5` hold command.

After a hidden 3D rotation, a multi-axis command can have an individual
component outside the upstream normalized Box even though its Euclidean norm
is unchanged. `RoboCasaOSCController` therefore rescales each transformed
translation or rotation vector by its infinity norm only when necessary. This
preserves direction while enforcing the official native action bound.

The model input projection includes only:

- The upstream public task instruction from
  `annotation.human.task_description`.
- Public control semantics and action schema.
- RGB from `robot0_agentview_left`, `robot0_agentview_right`, and `robot0_eye_in_hand`.
- The model's own normalized action and reflection history.

The two `robot0_agentview` cameras are attached to the mobile robot base. Their
image directions therefore move with the base and must not be described as
fixed MuJoCo world directions. Their pinned camera quaternions do provide stable
base-relative directions: base `+y` projects approximately image-left, base
`+z` image-up, and base `+x` mainly along viewing depth with an upward image
component. The left and right views give base `+x` a small image-left and
image-right component, respectively. `robot0_eye_in_hand` moves with the wrist
and therefore has no fixed base-axis projection. The prompt publishes these
static control facts without exposing a dynamic camera or robot pose.

It excludes observation state, reward, success, distances, object or end-effector poses, native simulator actions, seeds, camera randomization, and concrete permutation, sign, or SO(3) samples. In the perturbed condition the prompt says that hidden controls exist and remain fixed for the episode, but does not reveal their values.

## Environment

Use the isolated uv backend:

```bash
python3 scripts/manage_uv_env.py create robocasa
python3 scripts/manage_uv_env.py smoke robocasa
```

The `robocasa` runner sets `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` for
offscreen API benchmarks as well as runtime smoke tests. Run benchmark commands
through `manage_uv_env.py` so this backend environment contract is applied.

Pinned dependencies in `envs/uv/robocasa.in`:

- RoboCasa commit `8f3c96ec8d1bfcd8126cad2bca887da98d30e997` (`v1.0` / RoboCasa365 release).
- robosuite is resolved by the uv lock from PyPI, then replaced during `create robocasa` by an editable source checkout pinned to commit `85abee228d1c43ab1939bce33028099945d453b4`. The upstream README asks users to install robosuite from `master`; this commit makes that moving instruction reproducible.
- `numpy==2.2.5` and `mujoco==3.3.1`, matching RoboCasa v1.0 import assertions.

`scripts/manage_uv_env.py lock robocasa` uses the `x86_64-manylinux_2_31` target for this backend because `lerobot==0.3.3` requires `rerun-sdk>=0.21,<0.23`, and the compatible `rerun-sdk==0.22.1` wheel is published as `manylinux_2_31_x86_64`. It also uses the CPU Torch backend to avoid CUDA wheels for this CPU smoke.

`envs/uv/robocasa-excludes.in` excludes `mink` from resolution. robosuite declares `mink==0.0.5`, whose `numpy<2` requirement conflicts with RoboCasa v1.0's `numpy==2.2.5`; RoboCasa's selected kitchen runtime does not use `mink` directly.

RoboCasa v1.0's built wheel omits `robocasa/scripts`, but `robocasa/utils/env_utils.py` imports `robocasa.scripts.dataset_scripts.playback_dataset_hdf5` at package import time. The create step therefore clones the same fixed RoboCasa commit into `.venvs/_sources/robocasa` and installs it editable with `--no-deps` after syncing the lock.

## Current Setup Status

As of 2026-07-29:

- `python3 scripts/manage_uv_env.py lock robocasa` completes and writes `envs/uv/locks/robocasa.lock.txt`.
- `python3 scripts/manage_uv_env.py create robocasa` completes, including the fixed RoboCasa and robosuite editable source installs.
- The pinned RoboCasa environment passes the complete backend and MLLM suite: `11 passed, 0 skipped`. Six warning instances come from Gymnasium reporting that reset/step observations do not match the upstream declared observation space; the returned RGB and actions are valid.
- `python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml` completes one random-agent episode: `average_steps=100.0`, `success_rate=0.0`.
- `python3 scripts/manage_uv_env.py run robocasa -- python scripts/play_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml --set ui.max_steps=0` starts the shared Pygame worker runtime, resets RoboCasa, extracts the configured camera frame, and exits cleanly.

The previous robosuite source compatibility blocker is resolved by replacing PyPI `robosuite==1.5.2` with source commit `85abee228d1c43ab1939bce33028099945d453b4`. PyPI still lets uv resolve quickly, but its `ManipulationEnv.__init__` does not accept RoboCasa's `load_model_on_init=False` call. The upstream RoboCasa README says to install robosuite from `master`; this branch pins the current `master` head to make that instruction reproducible.

Verified command sequence:

```bash
python3 scripts/manage_uv_env.py create robocasa
python3 scripts/manage_uv_env.py run robocasa -- python -m robocasa.scripts.setup_macros
python3 scripts/manage_uv_env.py run robocasa -- python scripts/download_robocasa_lightwheel_assets.py --type all
python3 scripts/manage_uv_env.py run robocasa -- python -c "import builtins; builtins.input=lambda prompt='': 'y'; from robocasa.scripts.download_kitchen_assets import download_kitchen_assets; download_kitchen_assets(['tex', 'tex_generative'])"
MLLM_TELEOP_REQUIRE_ROBOCASA=1 python3 scripts/manage_uv_env.py smoke robocasa
```

The random-agent benchmark is expected to have `success_rate=0.0`; the validation claim is reset/step/render/logging, not task success.

Upstream references:

- RoboCasa repository: https://github.com/robocasa/robocasa
- RoboCasa v1.0 release: https://github.com/robocasa/robocasa/releases/tag/v1.0
- robosuite v1.5.2 tag: https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.2

## Assets

RoboCasa assets are about 10 GB. They are not downloaded by tests or by importing this package.

After creating the uv environment, run:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python -m robocasa.scripts.setup_macros
python3 scripts/manage_uv_env.py run robocasa -- python scripts/download_robocasa_lightwheel_assets.py --type all
python3 scripts/manage_uv_env.py run robocasa -- python -c "import builtins; builtins.input=lambda prompt='': 'y'; from robocasa.scripts.download_kitchen_assets import download_kitchen_assets; download_kitchen_assets(['tex', 'tex_generative'])"
```

The verified downloader flow fetches zip files from Hugging Face datasets and extracts them under the installed `robocasa` package:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python -c "import robocasa; print(robocasa.__path__[0] + '/models/assets')"
```

The runtime smoke test skips when the package, compatible robosuite source, or assets are missing. Set `MLLM_TELEOP_REQUIRE_ROBOCASA=1` to make those failures hard failures during backend validation.

RoboCasa v1.0 still expects monolithic Lightwheel zip files, but the NVIDIA Hugging Face dataset now publishes split zip files under `fixtures_lightwheel/` and `objects_lightwheel/`. `scripts/download_robocasa_lightwheel_assets.py` pins the current dataset revision `420a04af939c34873e6839a586b70844baf28aab` and extracts those split archives into the paths expected by the pinned RoboCasa source.

For `generative_textures.zip`, Hugging Face's Xet download path stalled behind the local Clash proxy. `scripts/manage_uv_env.py run robocasa` sets `HF_HUB_DISABLE_XET=1` and removes `ALL_PROXY/all_proxy`, while keeping HTTP/HTTPS proxy variables:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python -c "import builtins; builtins.input=lambda prompt='': 'y'; from robocasa.scripts.download_kitchen_assets import download_kitchen_assets; download_kitchen_assets(['tex_generative'])"
```

## Validation

Run tests that do not require assets:

```bash
python3 -m pytest -q tests/test_robocasa_integration.py -rs
```

Run the actual backend and four-step Fake Client checks only after installing the pinned source and assets:

```bash
MLLM_TELEOP_REQUIRE_ROBOCASA=1 python3 scripts/manage_uv_env.py smoke robocasa
python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml
python3 scripts/manage_uv_env.py run robocasa -- python scripts/play_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml --set ui.max_steps=0
```

The Fake Client tests run both MLLM configs against the real simulator without reading `.env.model-api.local` or making a network request. They verify request history counts, prompt v2, private-state exclusion, native Dict shapes and bounds, all three RGB streams, non-black PNG artifacts, trajectory reflections, and the run manifest.

## MLLM Control Timing

The 100-decision MLLM protocol uses the upstream PandaOmron accumulated-goal
controller. A zero motion command holds the desired controller goal; it is not
an instantaneous brake, so the arm can continue moving toward the previous
goal. The binary gripper command is also applied incrementally: repeated close
commands may be required before the fingers are visually closed.

These timing properties explain why full-scale discrete corrections can
overshoot and why a one-step close command may not produce a grasp. They do not
change the verified channel order, base-frame axis signs, gripper polarity, or
native action scale. The current baseline and perturbed MLLM pair intentionally
keeps one unchanged prompt contract; future prompt revisions that expose these
timing details must be versioned and evaluated as a separate protocol.
