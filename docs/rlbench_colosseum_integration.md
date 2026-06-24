# RLBench And Colosseum Integration

RLBench is the first CoppeliaSim/PyRep backend for this benchmark route. Colosseum is treated as a robustness extension on the same family of runtime dependencies, but it is not wired as a simulator type until the base RLBench smoke test is verified against a real local CoppeliaSim install.

## Verification Status

Implemented in this repository:

- `simulator.type: rlbench`
- `task.type: rlbench_generic`
- `controller.type: rlbench_delta_eef`
- Example config: `configs/rlbench_reach_target_delta_eef.yaml`
- uv backend: `rlbench`

Not claimed yet:

- `simulator.type: colosseum`
- Colosseum perturbation configs
- Any Colosseum runtime smoke result

The blocker is explicit: Colosseum builds on the same CoppeliaSim/PyRep/RLBench route, and this worktree has not verified a real RLBench `reset -> step -> render -> close` smoke run on a host with CoppeliaSim. Do not add Colosseum runtime claims until that passes.

## External Prerequisites

uv manages only Python packages for the `rlbench` backend. It does not install or validate the CoppeliaSim system binary.

RLBench upstream documents CoppeliaSim v4.1.0 with PyRep. Use a matching Linux CoppeliaSim archive and set:

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

For headless rendering, RLBench/PyRep still need a working OpenGL path. On GPU hosts this usually means an NVIDIA driver, an X server such as `DISPLAY=:99`, and host-level X/GPU setup outside uv. On desktop hosts, a normal display-backed OpenGL context is enough. If these pieces are missing, the smoke test should skip or fail with an actionable CoppeliaSim/PyRep message rather than pretending the backend is usable.

## Environment

Create the Python backend:

```bash
python3 scripts/manage_uv_env.py create rlbench
```

Run the focused smoke:

```bash
python3 scripts/manage_uv_env.py smoke rlbench
```

Run only the tests:

```bash
python3 scripts/manage_uv_env.py run rlbench -- python -m pytest -q tests/test_rlbench_colosseum_integration.py -rs
```

To make runtime launch failures fail the test instead of being reported as optional-backend skips:

```bash
MLLM_TELEOP_REQUIRE_RLBENCH=1 python3 scripts/manage_uv_env.py run rlbench -- python -m pytest -q tests/test_rlbench_colosseum_integration.py -rs
```

## Control Contract

The default config uses RLBench `ReachTarget` with a fixed action-mode choice:

```text
MoveArmThenGripper(
  EndEffectorPoseViaPlanning(),
  Discrete(),
)
```

The benchmark control path remains:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> RLBenchDeltaEEFController -> RLBenchSimAdapter -> task.step()
```

`RLBenchDeltaEEFController` emits an internal `RLBenchAction` containing delta end-effector position, optional delta RPY rotation, and a gripper scalar. `RLBenchSimAdapter` is the only layer that reads `obs.gripper_pose` and converts that delta command into RLBench's native absolute pose action:

```text
[x, y, z, qx, qy, qz, qw, gripper]
```

For `ReachTarget`, the initial config exposes only `dx`, `dy`, and `dz` virtual joysticks. The gripper is held open with `controller.params.default_gripper: 1.0` because the fixed RLBench action mode still expects a gripper value.

## Cameras

The first config uses the RLBench `front` RGB camera. Camera names must match RLBench observation names without the `_rgb` suffix:

- `front`
- `wrist`
- `left_shoulder`
- `right_shoulder`
- `overhead`

The observation provider configures RGB-only RLBench observations before environment launch and extracts camera frames from attributes such as `front_rgb`.

## Colosseum Plan

Colosseum should be added only after RLBench has a verified runtime smoke on the target host. The next minimal step is a perturbation config over the same route, not a parallel simulator abstraction:

1. Verify `configs/rlbench_reach_target_delta_eef.yaml` with real CoppeliaSim/PyRep.
2. Add the Colosseum Python dependency in a separate `rlbench-colosseum` backend or extend the documented backend name intentionally.
3. Add a single perturbation config using an official Colosseum task/variation.
4. Add smoke coverage that either runs `reset -> step -> render -> close` or fails loud with the exact missing Colosseum/CoppeliaSim prerequisite.

Until step 1 is done, tests contain an explicit skip documenting this blocker.
