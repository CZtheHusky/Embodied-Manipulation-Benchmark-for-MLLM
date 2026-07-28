# Simulator Backend Status

Last updated: 2026-07-14. Runtime results below were last verified on 2026-06-24.

This file records the environment-management and human-play status for simulator backends in `main`. It is not a replacement for each backend's integration document.

## Support Levels

- `uv`: uv lock plus normal package install.
- `uv_source`: uv lock plus pinned source checkouts installed after sync.
- `external_runtime`: uv-managed Python dependencies plus required host runtime, simulator binary, GPU driver, or display stack.
- `fallback_blocked`: uv is not currently reliable enough; use a documented conda or container fallback instead.

## Source Pinning Rule

Backends that depend on upstream source trees must record the source repository and exact 40-character commit SHA in the corresponding environment manager. Floating branches, tags, or unverified PyPI wheels are not enough for runtime verification. The create command fetches each source into `.venvs/_sources/`, checks out the pinned commit, verifies `HEAD`, and installs that checkout into the backend environment.

## Integrated In `main`

`Human play` means that a checked-in config uses the shared Pygame frontend and
`sim_adapter_worker` runtime, and that its startup/zero-step smoke has passed in the
documented environment. It does not mean every upstream task has a dedicated config.

| Backend | Environment management | Human play | Config / notes |
| --- | --- | --- | --- |
| `core` | `uv` | Infrastructure verified | Core smoke tests run through `scripts/manage_uv_env.py smoke core`; this is not a simulator task. |
| `gymnasium` | `uv` | Ready | `configs/reacher_delta_eef.yaml`, `configs/pusher_delta_eef.yaml`. |
| `dmcontrol` | `uv` | Ready | `configs/dmcontrol_example.yaml`. |
| `robosuite` | `uv` | Ready | `configs/robosuite_lift.yaml`. |
| `maniskill-cpu` | `uv` | Ready | `configs/maniskill_pickcube_delta_eef.yaml`; CPU runtime. |
| `maniskill-cu126` | `external_runtime` | Ready on compatible NVIDIA hosts | Uses the same ManiSkill config. uv manages Python dependencies; runtime requires a working NVIDIA driver and CUDA-capable Torch wheel. |
| `metaworld` | `uv` | Ready | `configs/metaworld_reach.yaml`. |
| `simpler-env` | `uv_source` | Ready with host GPU/display access | `configs/simpler_env_google_robot_pick_coke_can.yaml`. Installs pinned `mani_skill2_real2sim` commit `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` and `simpler_env` commit `06accaca93535902d408da4855f21cece12bceb7`. SAPIEN/Vulkan can fail in restricted sandboxes. |
| `calvin` | `uv_source` | Ready | `configs/calvin_delta_eef.yaml`. Installs pinned `calvin_env` commit `797142c588c21e76717268b7b430958dbd13bf48`. |
| `robocasa` | `uv_source` | Ready | `configs/robocasa_pick_place_counter_to_cabinet.yaml`. Pins RoboCasa commit `8f3c96ec8d1bfcd8126cad2bca887da98d30e997` and robosuite commit `85abee228d1c43ab1939bce33028099945d453b4`; required kitchen assets must be downloaded. |
| `rlbench` | Miniconda fallback | Ready with display/CoppeliaSim | `configs/rlbench_reach_target_delta_eef.yaml`. uv remains blocked, but the verified Miniconda environment pins RLBench commit `f214b6ecc177babce12d3ad13dd92049c824c1d9` and PyRep commit `076ca15c57f2495a4194da03565891ab1aaa317e`. |
| `libero` | Miniconda fallback | Ready | `configs/libero_spatial_task0.yaml`. uv remains blocked by the upstream dependency stack; the verified Miniconda environment pins LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. |

## Not Independently Ready

| Backend | Current state | Missing work |
| --- | --- | --- |
| RoboCasa365 | Covered only through the pinned RoboCasa v1.0/RoboCasa365 source release | No separate RoboCasa365 backend or task/config-level human-play entry has been added and verified. |
| Colosseum | Planned as an RLBench robustness extension | No pinned Colosseum checkout, independent backend/config, or human-play runtime verification exists yet. |

## Readiness Rule

A backend should not be marked ready until these are true:

1. Its support level is explicit in `scripts/manage_uv_env.py` or its fallback is explicit in `scripts/manage_conda_env.py`.
2. Any upstream source dependency is pinned to an exact commit and installed through the source checkout path.
3. The corresponding environment manager's `list` command shows the backend.
4. `create` and `smoke` pass on the target host. A fail-fast `blocked_reason` is explicit but not ready.
5. The backend imports remain lazy enough that `smoke core` passes without the simulator installed.
