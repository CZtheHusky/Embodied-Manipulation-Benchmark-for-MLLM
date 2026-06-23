# Agent Handoff

Prepared on 2026-06-23 after inspecting this directory, the main repository,
and local Gemini/Codex history.

## Directory Role

`Embodied-Manipulation-Benchmark-for-MLLM-Gemini1` is a sibling copy of the
main benchmark repository with a DeepMind Control Suite backend integration
attempt added on top.

Despite the directory name, this is not an integration with the Gemini model.
The current code changes are about `dm_control`.

## What Was Added Here

DMControl-specific files include:

- `configs/dmcontrol_example.yaml`
- `envs/environment-dmcontrol.yml`
- `docs/dmcontrol_integration.md`
- `tests/test_dmcontrol_integration.py`
- `src/mllm_teleop_benchmark/control/dmcontrol_controller.py`
- `src/mllm_teleop_benchmark/sim/dmcontrol/actions.py`
- `src/mllm_teleop_benchmark/sim/dmcontrol/adapter.py`
- `src/mllm_teleop_benchmark/sim/dmcontrol/observation.py`
- `src/mllm_teleop_benchmark/tasks/dmcontrol_task.py`
- `src/mllm_teleop_benchmark/ui/direct_dmcontrol_player.py`

Registry additions:

- `simulator.type: dmcontrol`
- `task.type: dmcontrol_generic`
- `controller.type: dmcontrol`
- observation provider branch for `dmcontrol`
- human-play backend `direct_dmcontrol`

The example config targets:

```text
domain_name: manipulator
task_name: bring_ball
```

## Intended Control Path

`DMControlController` translates benchmark joystick actions to a native
DMControl action tuple. It applies:

```text
JoystickAction -> JoystickMapping -> optional ControlFrameTransform -> scales -> DMControlAction
```

The example config uses dimensions:

```text
[dx, dy, dz, droll, gripper]
```

The controller requires explicit `controller.params.dimensions`; it does not
infer action layout from `env.action_spec()`.

Native action validation happens after controller translation at the simulator
boundary. `sim/dmcontrol/actions.py` converts `DMControlAction` to a numpy array,
checks it against `env.action_spec()` shape, rejects non-finite values, and clips
bounded actions. Both `DMControlSimAdapter` and the direct worker use this path.

## Current Implementation Notes

`DMControlSimAdapter` loads environments with `dm_control.suite.load(...)` and
sets `MUJOCO_GL` from config before importing the suite. It seeds
`env.task.random` when available, then uses `env.reset()` and `env.step()`.

`DMControlObservationProvider` renders configured cameras through
`sim.render(camera)`. `_apply_mujoco_camera_config(...)` now supports explicit
named fixed-camera model edits for `cam_pos`, `cam_quat`, and `cam_fovy`.
The shared `RandomizedMujocoCameraManager` still emits free-camera orbit fields
(`distance`, `azimuth`, `elevation`, `lookat`), which DMControl does not support
through this provider yet; that path raises `NotImplementedError` instead of
silently passing. The example config uses `camera.manager: fixed` and does not
randomize camera poses.

`DirectDMControlPlayer` runs DMControl in a separate worker process to avoid
OpenGL conflicts. This worker still directly calls `suite.load(...)` instead of
reusing `DMControlSimAdapter` and `DMControlObservationProvider`, but it shares
the same native action validation/clipping helper. It renders at the configured
camera spec size to avoid exceeding the DMControl offscreen framebuffer.

## History Evidence

The Gemini chat at:

```text
~/.gemini/tmp/embodied-manipulation-benchmark-for-mllm-gemini1/chats/session-2026-05-14T07-35-f54c9954.jsonl
```

ends with a claim that DMControl integration was completed, including conda
environment setup, adapter/controller/observation/task, direct manual play,
example config, and tests.

The 2026-06-23 repair pass removed the most serious overclaims. Fixed-camera
pose/FOV edits are implemented and tested; free-camera orbit randomization is
documented as unsupported for DMControl.

## Verification Run For This Handoff

Commands run:

```bash
python3 -m compileall -q src tests scripts
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-dmcontrol/bin/python -m pytest -q tests/test_dmcontrol_integration.py
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-dmcontrol/bin/python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

Results:

- DMControl target tests passed: `6 passed in 1.44s`.
- The example benchmark completed one random-policy episode: `average_steps: 250.0`, `num_episodes: 1`, `success_rate: 0.0`.
- A direct DMControl worker smoke produced an initial frame shape `(240, 320, 3)` and stepped once after clipping an intentionally out-of-bounds action.

The test file includes a short real benchmark episode when `dm_control`,
`mujoco`, and `numpy` are installed, so this is stronger than a pure mock test.
It is still not a full interactive manual-play validation.

## Known Risks And Gaps

- `RandomizedMujocoCameraManager` free-camera orbit randomization is unsupported
  for DMControl. Use explicit named fixed-camera pose/FOV edits or add a
  DMControl-specific camera manager.
- `DirectDMControlPlayer` bypasses `DMControlSimAdapter` and
  `DMControlObservationProvider`. This is currently justified as a GL isolation
  shortcut, with shared action validation/clipping to keep controller semantics
  aligned.
- The example camera names `fixed` and `hand` rendered in the test/benchmark
  environment, but other DMControl tasks need camera name validation.
- This directory is not merged into the main repository.

## Suggested Next Work

1. Decide whether manual play should fully reuse `DMControlSimAdapter` and
   `DMControlObservationProvider` inside the worker, or keep the current
   documented exception path.
2. Add a DMControl-specific camera randomization manager if free-camera orbit
   perturbations are required.
3. Run interactive manual play on a machine with a working display:

```bash
python scripts/play_benchmark.py --config configs/dmcontrol_example.yaml --backend direct_dmcontrol
```

4. If keeping this backend, port it into the main repo surgically and update
   `README.md`, `docs/environment_management.md`, `docs/manual_play.md`, and
   `docs/development_log.md`.
