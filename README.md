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
  tasks/        task interface and reach-target task
```

## Scope

The first version is deliberately minimal. It does not train low-level robot controllers, call MLLM APIs, or bind the benchmark to a specific simulator. Those integrations should be added behind the existing `SimAdapter`, `Agent`, `Controller`, and mapping interfaces.
