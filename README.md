# Embodied-Manipulation-Benchmark-for-MLLM

A Python skeleton for evaluating MLLMs as closed-loop embodied manipulation agents through a virtual joystick interface.

## Environment

This project is managed with uv-first backend environments. Each simulator has a separate uv lock and virtual environment so heavy simulator dependencies are not installed together:

```bash
python3 scripts/manage_uv_env.py create core
python3 scripts/manage_uv_env.py create gymnasium
python3 scripts/manage_uv_env.py create maniskill-cpu
python3 scripts/manage_uv_env.py create robosuite
python3 scripts/manage_uv_env.py create dmcontrol
python3 scripts/manage_uv_env.py create calvin
```

Run commands through the target backend environment:

```bash
python3 scripts/manage_uv_env.py run core -- python scripts/run_benchmark.py --config configs/dummy_reach.yaml
```

Existing conda env files under `envs/environment-*.yml` are retained as fallback during the migration.

More details are in `docs/environment_management.md`.

## Run the Dummy Benchmark

```bash
python3 scripts/manage_uv_env.py run core -- python scripts/run_benchmark.py --config configs/dummy_reach.yaml
```

The initial config runs a deterministic dummy reach-target task with:

- fixed camera metadata,
- fixed joystick mapping,
- EE delta pose control,
- scripted policy,
- JSON summary and JSONL trajectory logging.

## Test

```bash
python3 scripts/manage_uv_env.py smoke core
```

## Run Reacher

The first real MuJoCo-backed environment is a local Gymnasium environment:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/run_benchmark.py --config configs/reacher_delta_eef.yaml
```

For stable human evaluation of the local Reacher environment:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml
```

The default manual Reacher player uses a two-process backend: Pygame runs in the main process and MuJoCo/Gymnasium rendering runs in a worker process. This avoids local OpenGL context conflicts while still using the standard joystick parser, control interface, controller, metrics, and trajectory logger. The window shows a control-information panel above the environment RGB frame. The `--backend benchmark` option is still available for single-process debugging, but the direct backend is the tested manual-play path for Reacher.

Keyboard controls:

- `W/S`: Reacher delta x
- `A/D`: Reacher delta y
- `R`: reset
- `Esc` or `P`: quit

Reacher exposes two virtual joystick channels to both humans and MLLM policies. The public control panel and MLLM prompt disclose fixed visible mappings, and only hide exact details when mapping/frame complexity is enabled.

Config values can be overridden from the CLI for both benchmark and human play:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --set control_interface.mapping_strategy=permuted_sign_flipped
```

2D control-frame rotation can also be selected from the same interface:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --set control_interface.frame_transform=rotation_2d --set control_interface.params.frame_transform.angle_degrees=90
```

Random 2D rotation samples uniformly from SO(2). Random 3D rotation samples uniformly from SO(3) by default:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/play_benchmark.py --config configs/reacher_delta_eef.yaml --set control_interface.frame_transform=random_rotation_3d
```

More details are in `docs/manual_play.md` and `docs/reacher_integration.md`.

## Run ManiSkill

ManiSkill uses the `maniskill-cpu` uv environment by default:

```bash
python3 scripts/manage_uv_env.py run maniskill-cpu -- python scripts/run_benchmark.py --config configs/maniskill_pickcube_delta_eef.yaml
```

The initial ManiSkill config targets `PickCube-v1` with `pd_ee_delta_pose` control. Camera randomization is episode-level only; the camera view is fixed inside an episode. More details are in `docs/maniskill_integration.md`.

## Run robosuite

robosuite uses its own uv environment:

```bash
python3 scripts/manage_uv_env.py run robosuite -- python scripts/run_benchmark.py --config configs/robosuite_lift.yaml
```

The adapter sets `NUMBA_DISABLE_JIT=1` by default before importing robosuite to avoid the known numba cache failure in the pinned env.

## Run DMControl

DeepMind Control Suite uses its own uv environment:

```bash
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

The initial config uses `manipulator/bring_ball` through `dm_control.suite.load`. More details are in `docs/dmcontrol_integration.md`.

## Run CALVIN

CALVIN uses a pinned local editable checkout because its Hydra configs are required at runtime. `manage_uv_env.py create calvin` clones and checks out that upstream source under `.venvs/_sources/`.

```bash
python3 scripts/manage_uv_env.py create calvin
python3 scripts/manage_uv_env.py run calvin -- python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml
```

The initial config wraps `PlayTableSimEnv`, extracts `static` and `gripper` RGB observations, and runs a random-policy smoke episode. More details are in `docs/calvin_integration.md`.

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
  sim/maniskill/
               optional ManiSkill adapter and RGB observation provider
  sim/robosuite_adapter.py
               optional robosuite adapter
  sim/dmcontrol/
               optional DeepMind Control Suite adapter
  sim/calvin/
               optional CALVIN adapter
  tasks/        task interface and reach-target task
  ui/           human keyboard/Pygame evaluation interface
```

## Scope

The first version is deliberately minimal. It does not train low-level robot controllers, call MLLM APIs, or bind the benchmark to a specific simulator. Those integrations should be added behind the existing `SimAdapter`, `Agent`, `Controller`, and mapping interfaces.
