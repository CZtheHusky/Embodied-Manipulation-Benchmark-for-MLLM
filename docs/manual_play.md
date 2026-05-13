# Manual Play

Manual play is a debugging and human-evaluation tool for interacting with environments before an MLLM policy is connected.

The window is split into an information panel and an environment RGB view. The panel is generated from the same control-instruction builder used for MLLM prompts, so it should match the control interface metadata visible to the model.

## Reacher

Run:

```bash
conda activate mllm-teleop-benchmark
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml
```

The default config uses:

```yaml
ui:
  backend: direct_reacher
```

This backend starts two processes:

- main process: Pygame window, keyboard events, image display
- worker process: Gymnasium/MuJoCo env stepping and rendering

This split avoids OpenGL context conflicts between MuJoCo and Pygame. The main process still uses the benchmark action path: keyboard input becomes a standard joystick action from the configured control interface, then `ActionParser`, `ActionValidator`, mapping, and controller translate it into a simulator action for the worker.

The information panel shows task progress, control mode, joystick count, scale summary, mapping status, frame transform status, gripper availability, and the current joystick action.

## Keys

- `W/S`: end-effector x negative/positive
- `A/D`: end-effector y negative/positive
- `R`: reset
- `Esc` or `P`: quit

For Reacher, the configured control interface exposes only `joystick_0` and `joystick_1`; no gripper channel is shown or logged.
There are no diagonal macro keys by default. Human play sends the same single-step joystick schema that an MLLM policy must output.

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

## Known GL Failure Modes

Single-process Pygame + MuJoCo rendering can fail on X11 with:

- black window after the first frame
- black post-step render frames
- `EGL_BAD_ACCESS`
- `X_GLXMakeCurrent BadAccess`

Do not switch the default manual Reacher player back to a single-process renderer unless the target machine has been tested.

## Benchmark Backend

The script also supports:

```bash
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --backend benchmark
```

This path tries to run simulator rendering and Pygame in one process. It is useful for debugging architecture but is less robust for desktop play on machines with GL context conflicts.

## Logging

The default direct backend writes the normal episode files under `logging.output_dir`, including `trajectory.jsonl` when enabled. Manual transitions record keyboard-generated joystick actions, parsed actions, validation results, translated simulator actions, and worker-returned state.
