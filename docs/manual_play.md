# Manual Play

Manual play is a debugging and human-evaluation tool for interacting with environments before an MLLM policy is connected.

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
- worker process: Gymnasium/MuJoCo Reacher env and rendering

This split avoids OpenGL context conflicts between MuJoCo and Pygame.

## Keys

- `W/S`: end-effector x negative/positive
- `A/D`: end-effector y negative/positive
- `Q/E/Z/X`: diagonal moves
- `R`: reset
- `Esc` or `P`: quit

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

This path tries to use the full benchmark observation/controller path in one process. It is useful for debugging architecture but is less robust for desktop play on machines with GL context conflicts.
