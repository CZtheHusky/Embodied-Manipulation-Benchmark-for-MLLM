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
- The run manifest records Chat Completions, `action_with_reflection`, reasoning `low`, image history `3`, all action history, and prompt version `mllm_joystick_reflection_v2`.
- Sentinel state, reward, success, distance, object/EEF pose, simulator action, seed, and sampled hidden-control values did not enter the model request.

The validation command was:

```bash
PYTHONPATH=src .venvs/calvin/bin/python -m pytest tests/test_calvin_integration.py -q
```

It completed with `13 passed`, no skips, and four upstream Hydra compatibility warnings. No real model API or network access was used.

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

## MLLM Evaluation Configurations

Two configurations exercise the same task and differ only in the control condition:

- `configs/calvin_delta_eef_mllm_reflection.yaml`: fixed joystick mapping and identity control frame.
- `configs/calvin_delta_eef_mllm_reflection_perturbed.yaml`: episode-fixed `permuted_sign_flipped` mapping and `random_rotation_3d` frame transform.

Both configurations use:

- Chat Completions through the shared OpenAI-compatible client.
- `action_with_reflection`.
- reasoning effort `low`.
- `max_output_tokens: 192`.
- `image_window: 3` and `action_window: all`.
- the default-position `static` and `gripper` public RGB cameras.
- video disabled by default.

The MLLM configurations intentionally use the fixed CALVIN camera manager. The
legacy human-play configuration's broad static-camera randomization can sample
close-up views of the table that are non-black but do not show enough of the
workspace for visual control.

The public prompt explains translation and rotation axes, positive/negative motion, the Cartesian-relative controller frame, and the virtual gripper endpoints. For the perturbed condition, it states that hidden mapping and frame rules remain fixed for the whole episode and must be inferred from visual/action history. It does not reveal the sampled permutation, signs, rotation matrix, quaternion, simulator state, reward, success oracle, object or EEF poses, simulator action, or seed.

The task instruction is identical in both conditions:

```text
Pick up the object and place it in the drawer.
```

It contains task intent only; control semantics come from the shared public-control prompt.

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

Task success is not directly produced by the basic play-table smoke run. `CalvinTask` checks `success`, `is_success`, and `task_success` by default; real language-task evaluators may need a stricter task oracle if CALVIN's task annotations are connected later.

This remains the main evaluation-validity limitation for the new MLLM configurations. The RGB-to-model-to-action-to-simulator loop is ready, but the plain `PlayTableSimEnv` does not expose a language-task success key in the verified run. Meaningful CALVIN success rates require connecting the benchmark's task annotation/oracle layer; without it, episodes can run and be reviewed but normally terminate at `max_steps`.

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
