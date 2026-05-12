# Embodied-Manipulation-Benchmark-for-MLLM

A Python skeleton for evaluating MLLMs as closed-loop embodied manipulation agents through a virtual joystick interface.

## Environment

This project is intended to be managed with conda:

```bash
conda env create -f environment.yml
conda activate mllm-teleop-benchmark
```

In the local Codex environment, use the available shell helper if configured:

```bash
source ~/.bashrc
conda_activate
```

## Run the Dummy Benchmark

```bash
python scripts/run_benchmark.py --config configs/dummy_reach.yaml
```

The initial config runs a deterministic dummy reach-target task with:

- fixed camera metadata,
- fixed joystick mapping,
- EE delta pose control,
- scripted policy,
- JSON summary and JSONL trajectory logging.

## Test

```bash
pytest
```

## Run Reacher

The first real MuJoCo-backed environment is a local Gymnasium environment:

```bash
python scripts/run_benchmark.py --config configs/reacher_delta_eef.yaml
```

For stable human evaluation of the local Reacher environment:

```bash
python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml
```

The default manual Reacher player uses a two-process backend: Pygame runs in the main process and MuJoCo/Gymnasium rendering runs in a worker process. This avoids local OpenGL context conflicts. The `--backend benchmark` option is still available for exercising the generic benchmark runner path, but the direct backend is the tested manual-play path for Reacher.

Keyboard controls:

- `W/S`: Reacher delta x
- `A/D`: Reacher delta y
- `Q/E/Z/X`: diagonal moves
- `R`: reset
- `Esc` or `P`: quit

More details are in `docs/manual_play.md` and `docs/reacher_integration.md`.

## Package Layout

```text
src/mllm_teleop_benchmark/
  action/       joystick schema, parsing, validation
  agents/       policy interfaces and baseline policies
  config/       YAML loading and typed config dataclasses
  control/      controller interfaces and EE delta pose controller
  core/         runner, episode loop, shared runtime types
  evaluation/   metrics aggregation
  logging/      episode summary and trajectory logging
  mapping/      joystick mapping and control-frame transforms
  observation/  camera manager and observation provider
  sim/          simulator adapter interface and dummy simulator
  sim/gymnasium_envs/
               local Gymnasium/MuJoCo envs and adapters
  tasks/        task interface and reach-target task
  ui/           human keyboard/Pygame evaluation interface
```

## Scope

The first version is deliberately minimal. It does not train low-level robot controllers, call MLLM APIs, or bind the benchmark to a specific simulator. Those integrations should be added behind the existing `SimAdapter`, `Agent`, `Controller`, and mapping interfaces.
