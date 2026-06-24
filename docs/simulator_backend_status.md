# Simulator Backend Status

Last evaluated: 2026-06-24.

This file records the environment-management status for simulator backends. It is not a replacement for each backend's integration document; it is the quick check for whether a backend belongs in the current integration branch.

## Support Levels

- `uv`: uv lock plus normal package install.
- `uv_source`: uv lock plus pinned source checkouts installed after sync.
- `external_runtime`: uv-managed Python dependencies plus required host runtime, simulator binary, GPU driver, or display stack.
- `fallback_blocked`: uv is not currently reliable enough; use a documented conda or container fallback instead.

## Integrated In `integration/sim-backends-wave1`

| Backend | Support level | Status | Notes |
| --- | --- | --- | --- |
| `core` | `uv` | Verified | Core smoke tests run through `scripts/manage_uv_env.py smoke core`. |
| `gymnasium` | `uv` | Available | Existing uv lock and smoke command for Gymnasium/MuJoCo controller tests. |
| `dmcontrol` | `uv` | Available | Existing uv lock and smoke command for DMControl integration. |
| `robosuite` | `uv` | Available | Existing robosuite backend remains separate from the unresolved RoboCasa branch. |
| `maniskill-cpu` | `uv` | Available | CPU Torch backend. |
| `maniskill-cu126` | `external_runtime` | Available on CUDA 12.6 hosts | uv manages Python deps, but smoke requires a working NVIDIA driver and CUDA-capable Torch wheel. |
| `metaworld` | `uv` | Verified | Meta-World MT1 reach-v3 backend and smoke benchmark are merged. |
| `simpler-env` | `uv_source` | Verified with host GPU/display access | uv syncs the lock, then installs pinned `mani_skill2_real2sim` and `simpler_env` source checkouts. SAPIEN/Vulkan can fail in restricted sandboxes even when the environment is valid on the host. |
| `calvin` | `uv_source` | Available | uv syncs the lock, then installs pinned `calvin_env` source checkout editable so runtime config files are present. |

## Kept In Separate Branches

| Backend | Intended support level | Branch | Current status | Blocking risk / next action |
| --- | --- | --- | --- | --- |
| RoboCasa / RoboCasa365 | `uv_source` | `feat/sim-robocasa` (`2d5281f`) | Not merged | Environment creation reached the point of installing pinned sources, but forced smoke still hits a robosuite API mismatch around `load_model_on_init`. Resolve by pinning the exact upstream robosuite revision required by RoboCasa or adapting the backend only after that API contract is confirmed from upstream docs/source. |
| RLBench + Colosseum | `external_runtime` | `feat/sim-rlbench-colosseum` (`82c3486`) | Not merged | Requires CoppeliaSim/PyRep external runtime setup and source transfer. Keep blocked until lock/create/smoke can be reproduced on the target host with the simulator binary path documented. |
| LIBERO | `fallback_blocked` | `feat/sim-libero` (`108de40`) | Not merged | Dependency stack and benchmark-validity risk remain higher than the uv migration can absorb. Use a conda or container fallback until a narrow runtime-only uv lock is proven. |

## Merge Rule

A backend should not move from the separate-branch table into the integrated table until these are true:

1. Its support level is explicit in `scripts/manage_uv_env.py`.
2. `python3 scripts/manage_uv_env.py list` shows that support level.
3. `create` and `smoke` either pass on the target host or fail fast with a documented `blocked_reason`.
4. The backend imports remain lazy enough that `smoke core` passes without the simulator installed.
