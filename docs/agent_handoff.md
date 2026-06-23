# Agent Handoff

Prepared on 2026-06-23 after inspecting this directory, the main repository,
and local Gemini/Codex history.

## Directory Role

`Embodied-Manipulation-Benchmark-for-MLLM-Gemini` is a sibling copy of the main
benchmark repository with a CALVIN backend integration attempt added on top.

Despite the directory name, this is not an integration with the Gemini model.
The current code changes are about CALVIN, the manipulation benchmark also known
as "Composing Actions from Language and Vision".

## What Was Added Here

CALVIN-specific files include:

- `configs/calvin_delta_eef.yaml`
- `envs/environment-calvin.yml`
- `docs/calvin_integration.md`
- `tests/test_calvin_integration.py`
- `src/mllm_teleop_benchmark/control/calvin_controller.py`
- `src/mllm_teleop_benchmark/sim/calvin/adapter.py`
- `src/mllm_teleop_benchmark/sim/calvin/observation.py`
- `src/mllm_teleop_benchmark/tasks/calvin_task.py`
- `src/mllm_teleop_benchmark/ui/direct_calvin_player.py`

Registry additions:

- `simulator.type: calvin`
- `task.type: calvin_generic`
- `controller.type: calvin_delta_eef`
- `camera.manager: calvin_randomized`
- human-play backend `direct_calvin`

## Intended Control Path

The CALVIN controller maps benchmark joystick commands to CALVIN's expected
7-value delta EEF action:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

`CalvinDeltaEEFController` applies benchmark joystick mapping first, then the
shared control-frame transform, then scales dimensions, then maps benchmark
gripper values from `[0, 1]` into CALVIN-style open/closed values using
`gripper_closed_value` and `gripper_open_value`.

The default config exposes six motion joysticks plus a final direct gripper
joystick.

## Current Implementation Notes

`CalvinSimAdapter` tries to use `gymnasium.make(env_id)` after importing
`calvin_env`, with a fallback to `calvin_env.envs.play_table_env.PlayTableSimEnv`
when no `env_id` is set.

`CalvinObservationProvider` extracts images from CALVIN observations shaped like:

```text
obs["rgb_obs"]["rgb_static"]
obs["rgb_obs"]["rgb_gripper"]
```

It maps those to benchmark camera names `static` and `gripper`. If images are
not found in the native observation, it falls back to `sim.render(camera)`.

`RandomizedCalvinCameraManager` samples episode-level `position` and `look_at`
values. `CalvinSimAdapter` then tries to turn those into PyBullet view matrices
and patch the internal CALVIN camera object.

`DirectCalvinPlayer` reuses the shared `DirectBenchmarkPlayer` main loop and
runs CALVIN in a worker process, matching the general two-process rendering
pattern used after MuJoCo/Pygame OpenGL conflicts.

## History Evidence

The `~/.gemini/tmp/embodied-manipulation-benchmark-for-mllm-gemini` directory
does not contain a full chat JSONL. Its `logs.json` mostly records prompts about
understanding the repo, selecting further environments, and planning robosuite
integration. Current files are therefore stronger evidence for this directory
than the surviving Gemini history.

The main Codex history shows the user wanted new environment agents to preserve
strict boundaries between task, simulator, controller, observation, UI, and
logging, and to keep human play and MLLM play on the same control interface.

## Verification Run For This Handoff

Initial handoff verification had skipped the real simulator. Worker B then performed real CALVIN validation on 2026-06-23.

Commands run after the Worker B changes:

```bash
/home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python -m compileall -q src tests scripts
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python -m pytest -q tests/test_calvin_integration.py -rs
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml
```

Results:

- `compileall` passed.
- CALVIN tests passed against installed `calvin_env`: `4 passed`.
- The benchmark command completed one real CALVIN episode of 100 steps and reported:

```json
{
  "average_steps": 100.0,
  "num_episodes": 1,
  "success_rate": 0.0
}
```

The `0.0` success rate is from the random policy smoke run and does not indicate task-solver quality.

## Known Risks And Gaps

- `mllm-teleop-calvin` now exists locally and imports `calvin_env` from
  `envs/src/calvin-env`.
- `tests/test_calvin_integration.py` now uses dependency detection. Without
  `calvin_env`, the real smoke test skips; with it installed, reset/image
  extraction/step must pass.
- `envs/environment-calvin.yml` now installs `calvin_env` from editable VCS source
  at commit `797142c588c21e76717268b7b430958dbd13bf48`, not PyPI.
- CALVIN reset, step, render, and observation shapes were verified against that
  installed version.
- Camera randomization patches internal camera objects by convention. Confirm
  this works with the selected CALVIN/PyBullet camera implementation before
  claiming camera randomization is production-ready.
- Success detection checks common success keys, but the basic play-table smoke
  run did not provide a language-task success oracle. Add a task-specific
  success extractor before using CALVIN for scored language-task evaluation.
- This directory is not merged into the main repository.

## Suggested Next Work

1. Add or connect a CALVIN task oracle if scored language-task evaluation is required.
2. Verify the `calvin_randomized` camera patch changes the actual rendered static camera, not just the sampled config metadata.
3. Exercise manual play on a machine with display/input access:

```bash
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python scripts/play_benchmark.py --config configs/calvin_delta_eef.yaml --backend direct_calvin
```

4. If keeping this backend, port it into the main repo surgically and update
   `README.md`, `docs/environment_management.md`, `docs/manual_play.md`, and
   `docs/development_log.md`.
