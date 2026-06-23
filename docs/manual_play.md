# Manual Play

Manual play is a debugging and human-evaluation tool for interacting with environments before an MLLM policy is connected.

The window is split into an information panel and an environment RGB view. The panel is generated from the same control-instruction builder used for MLLM prompts, so it should match the control interface metadata visible to the model.

## Reacher

Run:

```bash
conda activate mllm-teleop-gymnasium
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml
```

The default config uses:

```yaml
ui:
  backend: direct_gymnasium
```

This backend starts two processes:

- main process: Pygame window, keyboard events, image display
- worker process: Gymnasium/MuJoCo env stepping and rendering

This split avoids OpenGL context conflicts between MuJoCo and Pygame. The main process still uses the benchmark action path: keyboard input becomes a standard joystick action from the configured control interface, then `ActionParser`, `ActionValidator`, mapping, and controller translate it into a simulator action for the worker.

The information panel shows task progress, control mode, joystick count, scale summary, mapping status, frame transform status, gripper availability, and the current joystick action.

For the direct Gymnasium backend, Pygame canvas size is derived from the first valid raw frame returned by `env.render()`:

- window width = raw frame width
- environment image display height = raw frame height
- total window height = raw frame height + `ui.info_panel_height`

`ui.window_width` and `ui.window_height` are fallback values used only when no valid initial frame is available. The raw frame returned by `env.render()` is not resized before debugging output; it is only scaled for display if a later frame has a different size from the initial frame.

## Keys

- `W/S`: end-effector x negative/positive
- `A/D`: end-effector y negative/positive
- `R`: reset
- `Esc` or `P`: quit

For Reacher, the configured control interface exposes only `joystick_0` and `joystick_1`; no gripper channel is shown or logged.
There are no diagonal macro keys by default. Human play sends the same single-step joystick schema that an MLLM policy must output.

## Pusher

Run:

```bash
conda activate mllm-teleop-gymnasium
python scripts/play_benchmark.py --config configs/pusher_delta_eef.yaml
```

Pusher exposes three virtual joysticks:

- `W/S`: `joystick_0`
- `A/D`: `joystick_1`
- `Q/E`: `joystick_2`

Fixed visible mappings show the exact joystick-to-dimension mapping in the panel and MLLM prompt. If mapping confusion or frame rotation is enabled by YAML or CLI override, the panel only indicates that such complexity is present and does not reveal exact permutations, sign flips, or frame rotation parameters.

## CLI Overrides

Both human play and benchmark evaluation support config overrides:

```bash
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --set control_interface.mapping_strategy=permuted
```

Overrides mutate the loaded benchmark config before either human or policy execution, so they are not human-player-only settings.

For example, to rotate the 2D control frame by 90 degrees:

```bash
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --set control_interface.frame_transform=rotation_2d --set control_interface.params.frame_transform.angle_degrees=90
```

`random_rotation_2d` samples an angle uniformly on SO(2), optionally constrained by `angle_range_degrees`. `random_rotation_3d` samples uniformly on SO(3) by default. Euler-angle ranges are available only by explicitly setting `sampling=euler_ranges`.

## ManiSkill PickCube

Run:

```bash
conda activate mllm-teleop-maniskill
python scripts/play_benchmark.py --config configs/maniskill_pickcube_delta_eef.yaml
```

The default config uses:

```yaml
ui:
  backend: direct_maniskill
```

This backend follows the same direct-player architecture as Gymnasium: the main process owns Pygame, keyboard input, benchmark action parsing, controller translation, episode logging, and debug files; the worker process owns ManiSkill environment creation, stepping, RGB extraction, and rendering fallback.

PickCube exposes six motion joysticks plus a final gripper joystick:

- `W/S`: `joystick_0`
- `A/D`: `joystick_1`
- `Q/E`: `joystick_2`
- `J/L`: `joystick_3`
- `K/I`: `joystick_4`
- `U/O`: `joystick_5`
- `G` or `Space`: toggle the final gripper joystick

The keyboard path still emits the same full joystick schema as an MLLM policy. Fixed visible mappings show exact joystick assignments. If mapping confusion, sign flips, or frame rotation are enabled, the panel only says that complexity is present and does not reveal the exact mapping, signs, or rotation.

ManiSkill camera randomization remains episode-level only. The direct player displays the first active RGB camera returned by the configured observation provider and falls back to `env.render()` only if the RGB observation does not contain an image.

## DMControl

Run:

```bash
conda activate mllm-teleop-dmcontrol
python scripts/play_benchmark.py --config configs/dmcontrol_example.yaml --backend direct_dmcontrol
```

The default config uses:

```yaml
ui:
  backend: direct_dmcontrol
```

The direct DMControl player keeps `dm_control` rendering in a worker process for GL isolation. The worker shares the same native action validation/clipping helper as `DMControlSimAdapter`, but it does not fully instantiate the adapter path.

## CALVIN

Run:

```bash
conda activate mllm-teleop-calvin
python scripts/play_benchmark.py --config configs/calvin_delta_eef.yaml --backend direct_calvin
```

The default config uses:

```yaml
ui:
  backend: direct_calvin
```

The direct CALVIN player reuses the benchmark parser, validator, mapping, controller, logging, and debug recorders in the main process. The worker owns CALVIN environment creation, reset, stepping, and RGB frame extraction through `CalvinObservationProvider`.

## Known GL Failure Modes

Single-process Pygame + MuJoCo rendering can fail on X11 with:

- black window after the first frame
- black post-step render frames
- `EGL_BAD_ACCESS`
- `X_GLXMakeCurrent BadAccess`

Do not switch the default direct Gymnasium player back to a single-process renderer unless the target machine has been tested.

## Benchmark Backend

The script also supports:

```bash
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --backend benchmark
```

This path tries to run simulator rendering and Pygame in one process. It is useful for debugging architecture but is less robust for desktop play on machines with GL context conflicts.

## Logging

The default direct backends write the normal episode files under `logging.output_dir/<run_id>/episode_0000`, including `trajectory.jsonl` when enabled. Manual transitions record keyboard-generated joystick actions, parsed actions, validation results, translated simulator actions, and worker-returned state.

When `ui.debug_frames: true`, the direct backend also saves raw environment render frames under:

```text
<logging.output_dir>/debug/<env_slug>/<control_slug>/human_keyboard/<run_id>/episode_0000/frames/env_render/front/
```

When `ui.debug_control_history: true`, every human control event is recorded under:

```text
<logging.output_dir>/debug/<env_slug>/<control_slug>/human_keyboard/<run_id>/episode_0000/control_events.jsonl
```

Each debug run also writes:

```text
<logging.output_dir>/debug/<env_slug>/<control_slug>/human_keyboard/<run_id>/manifest.json
```

For Pusher, a typical run path is:

```text
outputs/pusher_delta_eef/debug/mllmteleop_pusherdeltaeef_v0/eef_delta_pose_dx_dy_dz/human_keyboard/20260513-153012-a3f91c2b/
```

For step keys, each JSONL row includes the debug run id, actor, environment tag, control tag, source, control id, raw joystick action, controller output, expected EEF delta, EEF position before and after the step, actual EEF delta, EEF target delta, and whether the target was clamped by the environment. This recorder is environment-agnostic as long as the environment state exposes an EEF position field such as `fingertip_position`, `ee_position`, or `end_effector_position`.
