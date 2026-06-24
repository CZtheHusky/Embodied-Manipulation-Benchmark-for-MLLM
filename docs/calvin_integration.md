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
