# RLBench And Colosseum Integration

RLBench is the first CoppeliaSim/PyRep backend for this benchmark route. Colosseum is treated as a robustness extension on the same family of runtime dependencies, but it is not wired as a simulator type until the base RLBench smoke test is verified against a real local CoppeliaSim install.

## Verification Status

Implemented in this repository:

- `simulator.type: rlbench`
- `task.type: rlbench_generic`
- `controller.type: rlbench_delta_eef`
- Example config: `configs/rlbench_reach_target_delta_eef.yaml`
- MLLM configs:
  - `configs/rlbench_reach_target_mllm_reflection.yaml`
  - `configs/rlbench_reach_target_mllm_reflection_perturbed.yaml`
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

The current smoke result is:

- `tests/test_rlbench_colosseum_integration.py`: `15 passed, 1 skipped`
- `configs/rlbench_reach_target_delta_eef_smoke.yaml`: one benchmark episode, `average_steps=1.0`, `success_rate=0.0`
- `scripts/play_benchmark.py --config configs/rlbench_reach_target_delta_eef_smoke.yaml --set ui.max_steps=0`: shared Pygame worker runtime starts RLBench, extracts the `front` camera frame, and exits cleanly.

The remaining skip is the explicit Colosseum placeholder. RLBench is now verified,
including fixed and perturbed MLLM Fake Client loops. Colosseum builds on this route,
but its upstream source is not pinned or installed and `simulator.type: colosseum`
and official Colosseum perturbation configs are still not wired.

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

`RLBenchDeltaEEFController` emits an internal `RLBenchAction` containing delta end-effector position, optional delta RPY rotation, and a gripper scalar. `RLBenchSimAdapter` is the only layer that reads the latest `obs.gripper_pose` and converts that delta command into RLBench's native absolute world-frame planning action:

```text
[x, y, z, qx, qy, qz, qw, gripper]
```

For the MLLM `ReachTarget` configs, translation is the only active motion:

```text
delta_scaled = 0.02 m * mapped_translation
delta_world = episode_frame_transform(delta_scaled)
position_target = workspace_clamp(latest_observed_gripper_position + delta_world)
```

The adapter sends `position_target` with the latest observed quaternion as an
absolute world-frame pose to `EndEffectorPoseViaPlanning`. It applies no sign
compensation. `episode_frame_transform` is identity in the baseline, so `+1` on
`dx` adds `+0.02 m` to world x. In the perturbed condition, the episode-fixed
hidden mapping and frame rotation produce `delta_world`; the adapter still adds
that result directly, without another sign change, before workspace clamping.
Rotation stays inactive, and the gripper is held open with
`controller.params.default_gripper: 1.0` because the native action mode still
expects a gripper value.

Episode variation selection is deterministic. When a config does not provide
an explicit variation, the adapter selects `episode_seed % variation_count`
and calls RLBench `set_variation()` before reset. The selected variation stays
in evaluator state and is excluded from model input.

RLBench planning can reject an otherwise valid bounded command when no path is
available. The adapter treats RLBench `InvalidActionError` as a rejected
decision: it keeps the latest observation, does not advance CoppeliaSim, marks
`action_rejected: true` in evaluator state, and lets the episode continue.
Unexpected simulator exceptions still propagate and fail the run.

## MLLM Evaluation

Both MLLM configs use the same public task text:

```text
Move the robot end-effector to the visible target.
```

They use `action_with_reflection`, OpenAI-compatible Chat Completions, reasoning
effort `low`, three historical image turns, all textual action history, and the
public `front` RGB camera. The public control surface contains only `dx`, `dy`,
and `dz`; it does not expose rotation or gripper channels because `ReachTarget`
does not need them. A magnitude-1 mapped translation is `0.02 m` before the
optional hidden frame rotation.

The baseline exposes the fixed joystick mapping and identity frame. The
perturbed condition uses an episode-fixed hidden `permuted_sign_flipped`
mapping and `random_rotation_3d` transform over the same three translation
dimensions. The prompt describes the RLBench world Cartesian frame but never
includes the sampled permutation, signs, rotation matrix, target pose,
end-effector pose, task low-dimensional state, reward, success oracle,
variation index, seed, or native simulator action.

The local Fake Client validation runs four real CoppeliaSim steps for each
condition and verifies:

- the `front` camera is non-black `320x240` RGB
- native actions are finite absolute poses inside configured workspace bounds
- a no-simulator canary verifies direct `0.02 m` scaling, accumulation from the
  latest observed absolute gripper pose, and workspace clamping
- quaternion and gripper fields satisfy the RLBench action contract
- prompt v2, reflection history, and three-image-turn behavior
- hidden state and API-key sentinels do not enter requests or artifacts

This validates the benchmark chain, not task completion by a real model.

## Cameras

The first config uses the RLBench `front` RGB camera. Camera names must match RLBench observation names without the `_rgb` suffix:

- `front`
- `wrist`
- `left_shoulder`
- `right_shoulder`
- `overhead`

The observation provider configures RGB-only RLBench observations before environment launch and extracts camera frames from attributes such as `front_rgb`.

The `front` camera pose belongs to the RLBench TTT scene. The benchmark does not
currently expose runtime front-camera extrinsics to the model, so neither the
prompt nor this integration assigns world `+x`, `+y`, or `+z` to image
left/right/up/down. Such an image-axis claim requires runtime extrinsics rather
than inference from the camera name or a rendered frame.

## Colosseum Plan

RLBench now has a verified runtime smoke and an internal hidden-control
perturbation condition. The next Colosseum-specific work is:

1. Select and pin an upstream Colosseum source commit.
2. Add it to a separate Miniconda backend or deliberately extend the RLBench backend.
3. Wire one official Colosseum task and perturbation through the existing RLBench adapter path where its API contract permits.
4. Add `reset -> step -> render -> close` coverage and compare the official perturbation metadata against the public-input policy.

Until those steps are done, tests contain an explicit skip documenting the
missing Colosseum source/runtime integration.
