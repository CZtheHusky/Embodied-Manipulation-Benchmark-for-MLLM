# RLBench And Colosseum Integration

RLBench is the first CoppeliaSim/PyRep backend for this benchmark route. Colosseum is treated as a robustness extension on the same family of runtime dependencies, but it is not wired as a simulator type until the base RLBench smoke test is verified against a real local CoppeliaSim install.

## Verification Status

Implemented in this repository:

- `simulator.type: rlbench`
- `task.type: rlbench_generic`
- `controller.type: rlbench_delta_eef`
- Example config: `configs/rlbench_reach_target_delta_eef.yaml`
- MLLM baseline: `configs/rlbench_reach_target_mllm_reflection.yaml`
- MLLM perturbation: `configs/rlbench_reach_target_mllm_reflection_perturbed.yaml`
- Miniconda fallback backend: `rlbench`

Not claimed yet:

- `simulator.type: colosseum`
- Colosseum perturbation configs
- Any Colosseum runtime smoke result

Verified on 2026-06-24 with display-backed CoppeliaSim:

```bash
python3 scripts/manage_conda_env.py install-coppeliasim
python3 scripts/manage_conda_env.py create rlbench
python3 scripts/manage_conda_env.py smoke rlbench
```

The smoke result was:

- `tests/test_rlbench_colosseum_integration.py`: `8 passed, 1 skipped`
- `configs/rlbench_reach_target_delta_eef_smoke.yaml`: one benchmark episode, `average_steps=1.0`, `success_rate=0.0`
- `scripts/play_benchmark.py --config configs/rlbench_reach_target_delta_eef_smoke.yaml --set ui.max_steps=0`: shared Pygame worker runtime starts RLBench, extracts the `front` camera frame, and exits cleanly.

The remaining skip is the explicit Colosseum placeholder. Colosseum builds on this now-verified RLBench/CoppeliaSim route, but `simulator.type: colosseum` and perturbation configs are still not wired.

## External Prerequisites

The Miniconda fallback manages Python packages and pinned source checkouts. It also provides a helper to install CoppeliaSim v4.1.0 under `/home/caozhe/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`.

RLBench upstream documents CoppeliaSim v4.1.0 with PyRep. Use a matching Linux CoppeliaSim archive and set:

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

The verified run used the existing desktop display socket (`DISPLAY=:1`) rather than headless mode. On this host, display-backed OpenGL is the supported path; headless GPU validation is not claimed.

## Environment

The fallback source pins are:

- PyRep: `https://github.com/stepjam/PyRep.git@076ca15c57f2495a4194da03565891ab1aaa317e`
- RLBench: `https://github.com/stepjam/RLBench.git@f214b6ecc177babce12d3ad13dd92049c824c1d9`

Create the Python backend:

```bash
python3 scripts/manage_conda_env.py create rlbench
```

Run the focused smoke:

```bash
python3 scripts/manage_conda_env.py smoke rlbench
```

This requires both the Python environment and the external CoppeliaSim/PyRep runtime. The helper sets `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH`, and falls back to `DISPLAY=:1` when that X socket exists.

Run the non-interactive manual-play smoke:

```bash
python3 scripts/manage_conda_env.py run rlbench -- python scripts/play_benchmark.py --config configs/rlbench_reach_target_delta_eef_smoke.yaml --set ui.max_steps=0
```

Run only the tests:

```bash
python3 scripts/manage_conda_env.py run rlbench -- python -m pytest -q tests/test_rlbench_colosseum_integration.py -rs
```

To make runtime launch failures fail the test instead of being reported as optional-backend skips:

```bash
MLLM_TELEOP_REQUIRE_RLBENCH=1 python3 scripts/manage_conda_env.py run rlbench -- python -m pytest -q tests/test_rlbench_colosseum_integration.py -rs
```

The older uv route is not the verified path for this backend. CoppeliaSim is an external binary, PyRep needs a source build, and the conda flow is the reproducible route currently proven on this host.

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

The MLLM configs expose the same three world-frame translation dimensions plus
one final binary gripper joystick. The public contract is `0=open` and
`1=closed`; the RLBench controller maps those commands to the native
`Discrete` action values `1=open` and `0=closed`. This keeps an all-zero
recovery action physically stable while preserving RLBench's native semantics.
The motion scale is 2 cm per non-zero command.

The public controller frame is the RLBench world-frame x, y, and z axes used by
the end-effector pose planner. `ReachTarget` therefore has a genuine 3D
Cartesian translation contract, so the perturbed config uses
`random_rotation_3d`, not a planar transform. The baseline uses
`fixed + identity`. The perturbed config uses
`permuted_sign_flipped + random_rotation_3d`; only the three motion channels
are permuted or sign-flipped, while the final gripper channel remains fixed.

## MLLM Evaluation

Both MLLM configs use:

- Chat Completions through the shared OpenAI-compatible client
- `action_with_reflection`
- reasoning effort `low`
- `max_output_tokens: 192`
- the current shared reflection prompt contract
- the current `front` RGB image plus three historical RGB images
- all prior normalized action/reflection text
- review video disabled by default

The task instruction is identical in both conditions and only asks the agent to
move the end-effector to the visible target. The public prompt explains the
world XYZ translation axes, positive/negative/zero motion, and the gripper
endpoints. In the perturbed condition it tells the model that fixed hidden
mapping and frame complexity must be inferred from public RGB and its own
actions, but it does not reveal the sampled permutation, signs, rotation
matrix, quaternion, or seed.

The request projection excludes RLBench state, reward, success, target or
end-effector pose, distance, task low-dimensional state, joint positions,
variation index, and the native simulator action. These values may remain in
evaluator-only trajectory data but are never included in the model request.

The adapter seeds the pinned RLBench release's process-global NumPy RNG before
variation selection and reset. Equal benchmark episode seeds therefore produce
equal variation, description, and target placement across the two control
conditions. After reset, the adapter performs one non-stepping
`TaskEnvironment.get_observation()` refresh. This avoids a CoppeliaSim first
capture that can contain only the table and background before robot/task
geometry appears.

Run the dedicated integration suite on the verified display route:

```bash
DISPLAY=:1 MLLM_TELEOP_REQUIRE_RLBENCH=1 \
  python3 scripts/manage_conda_env.py run rlbench -- \
  python -m pytest -q tests/test_rlbench_colosseum_integration.py -rs
```

The MLLM tests replace the HTTP client with a deterministic Fake Client and run
each control condition for four real simulator steps. They do not read
`.env.model-api.local`, contact a model endpoint, or require network access.
They verify request history, valid reflection actions, native 8D pose/gripper
actions, manifest fields, request privacy, and non-black `240x320` RGB PNGs.

Verified on 2026-07-29 with `DISPLAY=:1`:

- RLBench dedicated suite: `15 passed, 1 skipped`
- Core suite: `245 passed, 39 skipped`
- Baseline and perturbed step-0 evaluator state and RGB were identical for the
  same episode seed
- Both four-step RGB sequences contained the robot and task objects without
  black or incomplete frames
- The one dedicated skip is the explicit Colosseum placeholder

CoppeliaSim can emit `QObject::~QObject: Timers cannot be stopped from another
thread` during shutdown. It was non-fatal in this run; all three sequential
CoppeliaSim launches closed and the test process exited successfully.

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

1. Add the Colosseum Python dependency in a separate conda-managed backend or extend the documented backend name intentionally.
2. Pin the upstream Colosseum source commit before installing it.
3. Add a single perturbation config using an official Colosseum task/variation.
4. Add smoke coverage that either runs `reset -> step -> render -> close` or fails loud with the exact missing Colosseum/CoppeliaSim prerequisite.

Until those steps are done, tests contain an explicit skip documenting the Colosseum blocker.
