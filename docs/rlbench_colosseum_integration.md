# RLBench And Colosseum Integration

RLBench and Colosseum are separate simulator backends that share the benchmark's
RLBench action, controller, task, and observation contracts. They use independent
Miniconda environments and independent RLBench source checkouts because Colosseum
requires an older official RLBench pin.

## Verification Status

Implemented in this repository:

- `simulator.type: rlbench`
- `simulator.type: colosseum`
- `task.type: rlbench_generic`
- `controller.type: rlbench_delta_eef`
- RLBench config: `configs/rlbench_reach_target_delta_eef.yaml`
- Colosseum config:
  `configs/colosseum_close_box_manip_obj_color_smoke.yaml`
- MLLM configs:
  - `configs/rlbench_reach_target_mllm_reflection.yaml`
  - `configs/rlbench_reach_target_mllm_reflection_perturbed.yaml`
  - `configs/colosseum_close_box_mllm_reflection.yaml`
  - `configs/colosseum_close_box_mllm_reflection_perturbed.yaml`
- Miniconda fallback backends: `rlbench`, `colosseum`

The Colosseum route uses official `EnvironmentExt`, task `close_box`, and
`spreadsheet_idx=2`, whose official variation name is `manip_obj_color`.

Verified on 2026-06-24 with display-backed CoppeliaSim:

```bash
python3 scripts/manage_conda_env.py install-coppeliasim
python3 scripts/manage_conda_env.py create rlbench
python3 scripts/manage_conda_env.py smoke rlbench
```

The RLBench smoke result was:

- `tests/test_rlbench_colosseum_integration.py`: core tests and the real
  reset/step/render smoke passed in the RLBench environment
- `configs/rlbench_reach_target_delta_eef_smoke.yaml`: one benchmark episode, `average_steps=1.0`, `success_rate=0.0`
- `scripts/play_benchmark.py --config configs/rlbench_reach_target_delta_eef_smoke.yaml --set ui.max_steps=0`: shared Pygame worker runtime starts RLBench, extracts the `front` camera frame, and exits cleanly.

The independent Colosseum environment was runtime verified on 2026-07-30 with
display-backed CoppeliaSim. The forced smoke launched official
`EnvironmentExt`, reset `close_box` with `manip_obj_color`, checked a non-black
`320x240` front RGB frame, executed one end-effector action, and shut down
cleanly. `MLLM_TELEOP_REQUIRE_COLOSSEUM=1` was set, so runtime failures could
not be converted to skips.

## External Prerequisites

The Miniconda fallback manages Python packages and pinned source checkouts. It
also provides a helper to install CoppeliaSim v4.1.0 under
`/home/caozhe/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`.

RLBench upstream documents CoppeliaSim v4.1.0 with PyRep. Use a matching Linux CoppeliaSim archive and set:

```bash
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

The verified run used the existing desktop display socket (`DISPLAY=:1`) rather than headless mode. On this host, display-backed OpenGL is the supported path; headless GPU validation is not claimed.

## Environment

The existing RLBench backend pins remain:

- PyRep: `https://github.com/stepjam/PyRep.git@076ca15c57f2495a4194da03565891ab1aaa317e`
- RLBench: `https://github.com/stepjam/RLBench.git@f214b6ecc177babce12d3ad13dd92049c824c1d9`

The independent Colosseum backend pins are:

- Colosseum: `https://github.com/robot-colosseum/robot-colosseum.git@098a910ae1d490524adb49b8d50735b00a1a62de`
- RLBench: `https://github.com/stepjam/RLBench.git@7c3f425f4a0b6b5ce001ba7246354eb3c70555be`
- PyRep: `https://github.com/stepjam/PyRep.git@076ca15c57f2495a4194da03565891ab1aaa317e`
- CoppeliaSim: `4.1`

The source directories are named `colosseum`, `colosseum-rlbench`, and
`colosseum-pyrep`. They do not mutate the editable checkouts used by the
existing `rlbench` environment.

Create the Python backend:

```bash
python3 scripts/manage_conda_env.py create rlbench
```

Create the Colosseum backend:

```bash
python3 scripts/manage_conda_env.py create colosseum
```

Run the focused smoke:

```bash
python3 scripts/manage_conda_env.py smoke rlbench
```

Run the Colosseum smoke:

```bash
python3 scripts/manage_conda_env.py smoke colosseum
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

Colosseum uses a separate requirement flag. Without it, the external runtime
test skips when Colosseum, RLBench, PyRep, CoppeliaSim, or the display runtime is
unavailable. With it, the same condition fails:

```bash
MLLM_TELEOP_REQUIRE_COLOSSEUM=1 python3 scripts/manage_conda_env.py run colosseum -- python -m pytest -q tests/test_colosseum_integration.py -rs
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

The shared benchmark control path remains:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> RLBenchDeltaEEFController -> RLBenchSimAdapter -> task.step()
```

For `simulator.type: colosseum`, `ColosseumSimAdapter` replaces the final
adapter in that chain and launches official `EnvironmentExt`; the
`RLBenchAction` shape and controller behavior are unchanged.

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
expects a gripper value. The gripper is not controllable in these MLLM
conditions, and no joystick gripper channel is added.

Episode variation selection is deterministic. When a config does not provide
an explicit variation, the adapter selects `episode_seed % variation_count`
and calls RLBench `set_variation()` before reset. The selected variation stays
in evaluator state and is excluded from model input.

Neither pinned RLBench `TaskEnvironment` exposes a public success-query method,
and PyRep has no task-level success oracle. Both pinned RLBench implementations
do, however, retain the official native `Task` as `TaskEnvironment._task`.
`Task.success() -> (success, terminate)` is the authoritative oracle that each
pin's `TaskEnvironment.step()` calls after an action. The adapter calls that
same native oracle immediately after `TaskEnvironment.reset()` completes. This
does not send a no-op action or advance CoppeliaSim. The adapter fails loudly if
the pinned native oracle is unavailable; it does not infer reset success from
reward, termination, geometry, or observations.

All four MLLM configs set `task.params.reject_initial_success: true`.
`RLBenchTask.reset()` therefore rejects an episode before the first model call
when the native reset oracle reports success. `success` and `terminate` are
stored separately because native task failure conditions may terminate without
succeeding.

The baseline and perturbed configs in each backend intentionally have identical
benchmark seed, simulator config, and task config. Deterministic stub tests
prove that same-seed resets produce the same variation, descriptions, native
low-dimensional state, gripper pose, and reset success. Optional real-runtime
tests assert the same pairing, including the front RGB frame; they skip when
the pinned simulator runtime is unavailable. Those runtime pairing tests were
not executed while preparing this change.

RLBench planning can reject an otherwise valid bounded command when no path is
available. The adapter treats RLBench `InvalidActionError` as a rejected
decision: it keeps the latest observation, does not advance CoppeliaSim, marks
`action_rejected: true` in evaluator state, and lets the episode continue.
Unexpected simulator exceptions still propagate and fail the run.

## Official Colosseum Condition

On reset, `ColosseumSimAdapter` loads the official `close_box.yaml` and
`close_box.json` assets from the pinned editable Colosseum checkout. It verifies
that strategy entry 2 is `manip_obj_color`, applies the official
`get_spreadsheet_config` helper, replaces `env.seed` with the episode seed, and
passes the resulting `cfg.env` to:

```text
EnvironmentExt(
  ...,
  path_task_ttms=TASKS_TTM_FOLDER,
  env_config=cfg.env,
)
```

The adapter records task, perturbation, spreadsheet index, RLBench variation,
and episode seed under evaluator state `evaluator_metadata`. Model requests do
not serialize observation state or observation metadata; they receive only the
public task instruction, public control-interface description, action schema,
selected RGB images, and permitted action history.

The two Colosseum MLLM configs keep the official task and visual perturbation
identical while changing only the benchmark control condition. The baseline
publishes a fixed joystick mapping and identity frame. The perturbed config
uses an episode-fixed hidden permutation/sign mapping and hidden SO(3) control
rotation. Both expose only `front` RGB, the task text `Close the visible box
lid.`, three historical image turns, and the model's own textual action
history. The official perturbation name/index, RLBench variation, seed,
simulator state, reward, native action, and sampled hidden control transform
remain evaluator-only.

A no-simulator Colosseum canary sends a nonzero joystick command through the
perturbed mapping and controller. It independently reconstructs the scaled
controller-frame vector and episode-fixed rotation, verifies the resulting
world delta is nonzero, then verifies that `ColosseumSimAdapter._native_action`
adds exactly that world delta to the latest observed absolute pose. The native
gripper field remains fixed at open.

The shared invalid-action handler first imports
`rlbench.backend.exceptions.InvalidActionError`, as required by the official
Colosseum RLBench pin, and falls back to the legacy
`rlbench.task_environment` location for the existing RLBench backend.

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
dimensions. Its prompt states the order without disclosing sampled values: the
hidden mapping produces a controller-frame vector, component scaling is applied
there, and one hidden episode-fixed 3D rotation converts that vector into the
RLBench world-frame delta. The prompt never includes the sampled permutation,
signs, rotation matrix, target pose, end-effector pose, task low-dimensional
state, reward, success oracle, variation index, seed, or native simulator
action.

The local Fake Client validation runs four real CoppeliaSim steps for each
condition and verifies:

- the `front` camera is non-black `320x240` RGB
- native actions are finite absolute poses inside configured workspace bounds
- a no-simulator canary verifies direct `0.02 m` scaling, accumulation from the
  latest observed absolute gripper pose, and workspace clamping
- a Colosseum perturbed canary verifies a nonzero controller vector is rotated
  exactly once into the native world-frame pose delta
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

## Colosseum Runtime Status

The source pins, independent environment, adapter, official perturbation config,
registry route, evaluator metadata checks, and forced
`reset -> non-black render -> step -> close` test are verified on this host.
The two MLLM conditions still require their real API batches and final artifact
audits before model-evaluation readiness is claimed.
