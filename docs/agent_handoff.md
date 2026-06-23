# Agent Handoff

Prepared on 2026-06-23 after inspecting this directory, related Gemini/Codex
history, and the sibling `Embodied-Manipulation-Benchmark-for-MLLM-*`
directories.

## What This Repository Is

`Embodied-Manipulation-Benchmark-for-MLLM` is the main benchmark repository.
It is a Python framework for evaluating MLLMs as closed-loop embodied
manipulation agents through visual observations and a structured virtual
joystick interface.

The benchmark is not a low-level robot-controller training project and is not a
single-simulator demo. Its intended evaluation layer is:

```text
observation + task instruction + public control-interface metadata
-> agent output as virtual joystick JSON
-> parser / validator
-> joystick mapping
-> control-frame transform
-> backend-specific controller
-> SimAdapter.step()
-> logging and metrics
```

The design goal is to test visual spatial understanding, action grounding,
closed-loop correction, and robustness to future hidden control complexity such
as joystick permutation, sign flips, and rotated control frames.

## Current Main Capabilities

Implemented backend coverage in this directory:

- Dummy reach-target simulator for core pipeline tests.
- Local Gymnasium/MuJoCo Reacher delta-EEF style planar environment.
- Local Gymnasium/MuJoCo Pusher delta-EEF environment with damped least-squares
  IK and position actuators.
- Optional ManiSkill PickCube backend.
- Optional robosuite Lift backend pinned around `robosuite==1.4.1`.
- Optional DeepMind Control Suite backend for `manipulator/bring_ball`.
- Optional CALVIN backend for `PlayTableSimEnv` smoke evaluation.

Important shared components:

- `src/mllm_teleop_benchmark/config/schema.py`: typed config model. New configs
  must use `control_interface`; legacy top-level action/mapping config was
  removed.
- `src/mllm_teleop_benchmark/core/registry.py`: lazy backend construction.
  Heavy simulator imports should remain inside backend branches.
- `src/mllm_teleop_benchmark/mapping/`: joystick mapping and frame transforms.
- `src/mllm_teleop_benchmark/control/`: benchmark-to-backend controllers.
- `src/mllm_teleop_benchmark/ui/direct_player.py`: shared two-process human
  player loop.
- `src/mllm_teleop_benchmark/logging/debug_session.py`,
  `frame_debug.py`, and `control_debug.py`: debug artifact layout and recorders.

Human play and MLLM play are intended to use the same `control_interface`.
Human UI panels and MLLM prompts should reveal only whether mapping/frame
complexity exists, not exact permutations, signs, rotation matrices, or sampled
rotation parameters. If a gripper exists, it must stay as the final virtual
joystick and must not be permuted or sign-flipped.

## History Evidence

Relevant local histories inspected:

- `~/.codex/history.jsonl`: shows the original design request, implementation
  path, user constraints, and later requests for Gymnasium Reacher/Pusher,
  human play, hidden control complexity, debug logging, ManiSkill, robosuite,
  and backend environment isolation.
- `~/.gemini/tmp/embodied-manipulation-benchmark-for-mllm/logs.json`: shows
  debugging of the Reacher human-play black-window/OpenGL issue and environment
  setup work.
- `~/.gemini/tmp/embodied-manipulation-benchmark-for-mllm/chats/session-2026-05-14T10-37-b3d9709f.jsonl`:
  records a Gemini pass that created/verified separate Gymnasium, ManiSkill,
  and robosuite environments.
- `~/.codex/logs_2.sqlite` was checked with Python's sqlite library because the
  `sqlite3` CLI is not installed. It mostly surfaced current tool telemetry and
  was less useful than `~/.codex/history.jsonl` for older benchmark decisions.

The strongest current-state evidence is still the files in this repository.
Historical agent claims should be treated as context, not proof.

## Environment Policy

This repo now uses uv-first backend environments. Keep each simulator backend in
its own uv lock and virtual environment. The legacy conda files remain as
fallback during migration.

Current uv backend ids are:

- `core`
- `gymnasium`
- `maniskill-cpu`
- `maniskill-cu126`
- `robosuite`
- `dmcontrol`
- `calvin`

## Sibling Worktrees

Two sibling directories were used as source worktrees for backend recovery:

- `Embodied-Manipulation-Benchmark-for-MLLM-Gemini`: CALVIN integration attempt.
- `Embodied-Manipulation-Benchmark-for-MLLM-Gemini1`: DeepMind Control Suite
  integration attempt.

The DMControl and CALVIN backend implementations have now been ported into this
main directory after real-runtime smoke validation. The sibling directories
remain useful as historical context but should no longer be treated as the
source of truth.

## Verification Run For This Handoff

Historical commands run before the uv migration:

```bash
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-benchmark/bin/python -m pytest -q
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-dmcontrol/bin/python -m pytest -q tests/test_dmcontrol_integration.py
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-dmcontrol/bin/python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python -m pytest -q tests/test_calvin_integration.py -rs
PYTHONPATH=src /home/caozhe/miniconda3/envs/mllm-teleop-calvin/bin/python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml
```

Current uv equivalents:

```bash
python3 scripts/manage_uv_env.py smoke core
python3 scripts/manage_uv_env.py smoke gymnasium
python3 scripts/manage_uv_env.py smoke dmcontrol
python3 scripts/manage_uv_env.py smoke robosuite
python3 scripts/manage_uv_env.py smoke maniskill-cpu
python3 scripts/manage_uv_env.py smoke maniskill-cu126  # GPU host only
python3 scripts/manage_uv_env.py smoke calvin
```

Results:

- Historical main test suite passed before uv migration: `66 passed, 4 skipped`.
- Current uv core full pytest passed: `60 passed, 10 skipped`.
- Current uv Gymnasium full pytest passed: `65 passed, 5 skipped`.
- uv core smoke passed: `19 passed`.
- uv Gymnasium smoke passed: `6 passed`; a 2-step Reacher benchmark smoke completed with `success_rate: 0.0`.
- DMControl integration tests passed: `7 passed`; random-policy benchmark smoke completed one 250-step episode with `success_rate: 0.0`.
- robosuite uv smoke passed: `3 passed` after pinning `mujoco>=2.3,<3.0`.
- ManiSkill CPU uv smoke passed: `8 passed`.
- ManiSkill CUDA 12.6 env smoke passed on station with `torch==2.12.1+cu126`, CUDA 12.6, and NVIDIA GeForce RTX 2080. GPU checks must run outside the restricted command sandbox.
- CALVIN integration tests passed: `5 passed`; random-policy benchmark smoke completed one 100-step episode with `success_rate: 0.0`.
- Direct-worker reset/step smoke passed for DMControl and CALVIN without opening a Pygame display.

The smoke benchmark success rates are expected for random policies and do not
prove task-solving ability.

## Known Risks And Gaps

- `MLLMPolicy` is still a placeholder. There is no real API-backed MLLM policy
  implementation in this handoff scope.
- The targeted tests prove core control-interface/mapping/frame-transform
  behavior, not every optional simulator backend.
- DMControl free-camera orbit randomization is still unsupported and fails
  loudly.
- CALVIN camera randomization is best-effort against CALVIN internals and does
  not yet have a visual-change oracle.
- CALVIN task success still needs a stricter language-task oracle for real
  task-suite evaluation.
- History is incomplete: `~/.codex/history.jsonl` gives prompt-level context,
  while the durable Gemini files vary by directory and do not always preserve a
  full chat transcript.

## Best Next Steps

1. Preserve the project invariants:
   - no simulator-specific joystick parsing,
   - no hidden mapping/frame details in public prompt/UI metadata,
   - episode-level camera randomization only,
   - backend dependencies isolated by uv environment,
   - docs updated with every significant backend change.
2. Before real MLLM policy work, keep improving deterministic baselines,
   logging, replayability, and human-play parity.
