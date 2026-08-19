# CALVIN Integration

CALVIN (Composing Actions from Language and Vision) is an open-world benchmark for learning multi-step manipulation tasks.

## Verification Status

The backend is no longer only a mock/unit prototype in this directory.

Verified on 2026-06-23 with the `calvin` uv environment:

- `calvin_env` installs from `https://github.com/mees/calvin_env.git` at commit `797142c588c21e76717268b7b430958dbd13bf48`.
- The package reports version `0.0.1`.
- `pytest` runs a real `PlayTableSimEnv` reset, extracts `static` and `gripper` RGB images, and steps one zero-motion action.
- `scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml` runs one 100-step episode against real CALVIN. The random policy did not solve the task, so the observed `success_rate` was `0.0`.

The CALVIN source must be installed editable, not as a wheel, because the runtime Hydra configs live at the repository root `conf/` directory and are not included in the built wheel.

Verified again on 2026-07-29 with the pinned `.venvs/calvin` environment:

- The fixed/identity and perturbed MLLM configurations both load against the shared public-control contract.
- Each configuration completed a four-step real `PlayTableSimEnv` episode with a local Fake Client.
- Every model request used public `static` and `gripper` RGB images. The per-request image counts were `2, 4, 6, 8`, matching three historical turns plus the current turn.
- All eight model-input PNGs per condition were valid, non-black RGB images.
- Every parsed action was valid and translated to a finite seven-value native action with six bounded relative pose commands and a native gripper command in `{-1, 1}`.
- The run manifest records the `chat_completions` backend, optional Reflection, reasoning `low`, image history `3`, all action history, and prompt version `mllm_joystick_chat_completions_v10`.
- Sentinel state, reward, success, distance, object/EEF pose, simulator action, seed, and sampled hidden-control values did not enter the model request.
- Task success uses the pinned CALVIN `Tasks.get_task_info_for_set()` API. It requires `open_drawer`, then the configured target block's table/slider lift task, and finally `place_in_drawer`, deep-copying the state at each completed phase. Success remains sticky and never uses `push_into_drawer`.
- Both MLLM configurations enable `reject_initial_success`. The reset check uses absolute scene state: the drawer joint must be closed, the red block must not contact the drawer link, and the red block and robot must not contact each other.
- Legacy `PlayTableSimEnv.seed(seed)` resets now propagate its new RNG to the scene and every movable object. Real resets reproduce the same layout for the same seed and sample different layouts for different seeds.

The strict runtime validation command is:

```bash
MLLM_TELEOP_REQUIRE_CALVIN=1 \
python3 scripts/manage_uv_env.py run calvin -- \
  python -m pytest -q tests/test_calvin_integration.py -rs \
  --junitxml=/tmp/calvin-gate.xml
```

The gate must collect 29 tests and report zero skips, failures, and errors.
The MLLM loop uses a local Fake Client and does not call a real model API.
It also requires the system `ffmpeg` binary and verifies that each four-step
MLLM condition produces a non-empty review video.

## Design

The CALVIN integration follows the standard benchmark architecture:

- `CalvinSimAdapter`: Wraps `calvin_env` (PlayTableSimEnv).
- `CalvinDeltaEEFController`: Maps virtual joysticks to CALVIN's 7-DoF delta EEF action space (`[dx, dy, dz, dr, dp, dy, gripper]`).
- `CalvinObservationProvider`: Extracts `static` and `gripper` camera views from CALVIN observations.
- `CalvinTask`: Handles task instructions and success metrics.

## Control Interface

The default configuration (`configs/calvin_delta_eef.yaml`) exposes:
- 6 virtual joysticks for end-effector delta pose (`dx, dy, dz, droll, dpitch, dyaw`).
- 1 virtual joystick for the gripper. Benchmark values below `0.5` map to CALVIN close (`-1.0`), and values `>= 0.5` map to CALVIN open (`1.0`).

The native convention is verified in CALVIN's pinned `Robot.apply_action()` and `Robot.control_gripper()` implementation: native `+1` opens the gripper and native `-1` closes it. The public direct-gripper schema therefore advertises virtual `1` as open and virtual `0` as closed.

Pinned CALVIN treats the first three normalized values as PyBullet-world
Cartesian-relative target increments and multiplies native magnitude `1` by
`0.02 m`. The next three values are increments of the target pose's PyBullet
XYZ Euler parameters, not an axis-angle vector, and native magnitude `1` changes
the corresponding parameter by `0.05 rad`. These upstream gains are left
unchanged. Benchmark configs use normalized controller scales of `0.5` for
translation and `0.628318530717959` for rotation, producing the standardized
physical step of `0.01 m` and `0.0314159265358979 rad` (`1.8 degrees`).

## MLLM Evaluation Configurations

Two configurations exercise the same task and differ only in the control condition:

- `configs/calvin_delta_eef_mllm_chat_completions.yaml`: fixed joystick mapping and identity control frame.
- `configs/calvin_delta_eef_mllm_chat_completions_perturbed.yaml`: episode-fixed `permuted_sign_flipped` mapping and a `random_rotation_3d` frame transform over translation.

Both configurations use:

- Chat Completions through the shared OpenAI-compatible client.
- `chat_completions` with optional Reflection.
- reasoning effort `low`.
- `max_output_tokens: 192`.
- `image_window: 3` and `action_window: all`.
- the default-position `static` and `gripper` public RGB cameras.
- ten episodes and review video enabled by default.

The MLLM configurations intentionally use the fixed CALVIN camera manager. The
legacy human-play configuration's broad static-camera randomization can sample
close-up views of the table that are non-black but do not show enough of the
workspace for visual control.

The hidden 3D frame transform is applied only to `dx/dy/dz`. CALVIN's
`droll/dpitch/dyaw` are Euler-coordinate increments, so rotating that triple as
if it were an axis-angle vector would be mathematically invalid. The three Euler
channels still participate in the episode-fixed permutation and sign flips.
If the rotated translation would place any normalized component outside
`[-1, 1]`, the controller uniformly divides all three components by their
largest absolute component. This keeps the transformed direction and native
component bounds, but can shorten a diagonal step. This transform and bound
policy are public; the sampled permutation, signs, and rotation remain private.

The public prompt explains world-frame translation, XYZ Euler-parameter
increments, native step magnitudes, approximate fixed-static-camera projections,
the moving wrist-camera caveat, positive/negative motion, and the `0.5` virtual
gripper threshold. The shared dimension text identifies `droll/dpitch/dyaw` as
XYZ Euler-parameter increments rather than fixed-axis rotation-vector
components. For the perturbed condition, it states that hidden mapping and frame
rules remain fixed for the whole episode and must be inferred from visual/action
history. It does not reveal the sampled permutation, signs, rotation matrix,
quaternion, simulator state, reward, success oracle, object or EEF poses,
simulator action, or seed.

The task instruction is identical in human play and both MLLM conditions:

```text
Open the drawer, pick up the red block, and place the red block inside the drawer.
```

It contains task intent only; control semantics come from the shared public-control prompt. Oracle inputs, including object coordinates, contacts, simulator state, and the seed, remain private.

## Human Play

To interact with the CALVIN environment:

```bash
python3 scripts/manage_uv_env.py create calvin
python3 scripts/manage_uv_env.py run calvin -- python scripts/play_benchmark.py --config configs/calvin_delta_eef.yaml
```

The human play UI uses `ui.backend: pygame` and `ui.runtime: sim_adapter_worker`, which runs the CALVIN simulator in a separate worker process to avoid OpenGL context conflicts with Pygame while still using `CalvinSimAdapter` and `CalvinObservationProvider`.

## Cameras

CALVIN typically provides two main camera views:
1. `static`: A fixed global view of the workspace.
2. `gripper`: A wrist-mounted camera on the robot's end-effector.

The `CalvinObservationProvider` automatically maps these views to the benchmark's observation format.

The real CALVIN observation keys verified here are:

```text
obs["rgb_obs"]["rgb_static"] -> observation.images["static"]
obs["rgb_obs"]["rgb_gripper"] -> observation.images["gripper"]
obs["robot_obs"][:3] -> state["ee_position"] / state["tcp_position"]
```

## Runtime Compatibility Notes

The verified `PlayTableSimEnv` uses the old `gym` API:

- `reset()` returns a single observation.
- `step()` returns `(obs, reward, done, info)`.
- `render(mode="rgb_array")` returns the static RGB image.

`CalvinSimAdapter` normalizes these into the benchmark reset/step/render interfaces. It also supports newer Gymnasium-style `(obs, info)` reset and 5-value step results for compatibility with wrappers.

The plain play-table environment does not emit a language-task success key, so `CalvinTask` evaluates the pinned official task definitions directly. At commit `797142c588c21e76717268b7b430958dbd13bf48`, `conf/tasks/new_playtable_tasks.yaml` defines `open_drawer` as a positive `base__drawer` movement over `0.12`, separate target-specific lift tasks, and `place_in_drawer` as placement on `table/drawer_link`. The evaluator therefore uses three ordered phases:

1. Filter only `open_drawer` against reset info and deep-copy the first opened state.
2. Filter only `lift_red_block_table` and `lift_red_block_slider` against that opened state and deep-copy the first held state.
3. Filter only `place_in_drawer` against the held state.

Moving or pushing a block into the drawer, lifting it before the drawer phase, or opening the drawer without the later target lift cannot complete the task. The model receives only the unchanged task instruction and public control prompt; phase state and oracle inputs are evaluator-private.

## Reset Invariant

The pinned `config_data_collection.yaml` selects `calvin_scene_D`. Its scene configuration sets `base__drawer.initial_state` to `0`, and movable blocks with `initial_pos: any` are sampled only from the configured table and slider surfaces. `CalvinTask` additionally checks the authoritative reset info rather than treating a no-op task transition as an initial-state oracle:

- `scene_info.doors.base__drawer.current_state` must be `0` within `1e-6`.
- The red block must not contact the table's `drawer_link`, matching CALVIN's destination-contact semantics.
- Neither the robot's contacts nor the red block's contacts may contain the other body UID.

The baseline and perturbed MLLM configs set `reject_initial_success: true`, so a violated invariant raises before any model call. A real reset canary covers seeds `42..51`, rechecks these predicates, confirms the three-stage evaluator is false at reset, and requires non-black uint8 RGB from both `static` and `gripper`. Optional core-only runs skip when CALVIN, PyBullet, Hydra, or OmegaConf is unavailable; `MLLM_TELEOP_REQUIRE_CALVIN=1` converts every such missing dependency or worker runtime error into a hard gate failure.

Pinned CALVIN uses the old Gym fallback: `env.seed(seed)` replaces `env.np_random`, while the already-created scene and movable objects retain their previous RNG references. `CalvinSimAdapter` synchronizes those references before `reset()`. PyBullet can introduce sub-millimeter post-reset settling differences, so the real-environment regression checks same-seed layout reproducibility at a 1 mm tolerance while requiring different seeds to differ by more than 1 cm for at least one block.

## Camera Randomization

The integration supports episode-level camera randomization via `calvin_randomized` manager.

To enable it, use the following configuration:

```yaml
camera:
  manager: calvin_randomized
  cameras:
    - name: static
      params:
        position_ranges:
          - [0.5, 1.0]   # x range
          - [-0.2, 0.2]  # y range
          - [0.4, 0.8]   # z range
        look_at_ranges:
          - [-0.1, 0.1]
          - [-0.1, 0.1]
          - [0.0, 0.2]
```

At the start of each episode, a new camera position and look-at target are sampled. The `CalvinSimAdapter` then recomputes the PyBullet view matrix and injects it into the simulation before the environment reset.

This camera patching is best-effort against CALVIN internals. The smoke test verifies RGB extraction from the default `static_and_gripper` camera setup, not a quantitative camera-randomization oracle.
