# Simulator Backend Status

Last evaluated: 2026-06-24.

This file records the environment-management status for simulator backends. It is not a replacement for each backend's integration document; it is the quick check for whether a backend belongs in the current integration branch.

## Support Levels

- `uv`: uv lock plus normal package install.
- `uv_source`: uv lock plus pinned source checkouts installed after sync.
- `external_runtime`: uv-managed Python dependencies plus required host runtime, simulator binary, GPU driver, or display stack.
- `fallback_blocked`: uv is not currently reliable enough; use a documented conda or container fallback instead.

## Source Pinning Rule

Backends that depend on upstream source trees must record the source repository and exact 40-character commit SHA in `scripts/manage_uv_env.py`. Floating branches, tags, or unverified PyPI wheels are not enough for runtime verification. The create command fetches each source into `.venvs/_sources/`, checks out the pinned commit, verifies `HEAD`, and installs that checkout into the backend environment.

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
| `simpler-env` | `uv_source` | Verified with host GPU/display access | Installs pinned `mani_skill2_real2sim` commit `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` and `simpler_env` commit `06accaca93535902d408da4855f21cece12bceb7`. SAPIEN/Vulkan can fail in restricted sandboxes even when the environment is valid on the host. |
| `calvin` | `uv_source` | Available | Installs pinned `calvin_env` commit `797142c588c21e76717268b7b430958dbd13bf48` editable so runtime config files are present. |

## Kept In Separate Branches

| Backend | Intended support level | Branch | Current status | Blocking risk / next action |
| --- | --- | --- | --- | --- |
| RoboCasa / RoboCasa365 | `uv_source` | `feat/sim-robocasa` (`725dbe1`) | Not merged | RoboCasa is pinned to commit `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`, and robosuite is pinned to upstream `master` commit `85abee228d1c43ab1939bce33028099945d453b4`. The previous `load_model_on_init` blocker is resolved; required smoke now fails on missing downloaded kitchen assets such as `objects/lightwheel/stool/Stool017/model.xml`. |
| RLBench + Colosseum | `external_runtime` | `feat/sim-rlbench-colosseum` (`212684f`) | Not merged | RLBench is intended to use commit `f214b6ecc177babce12d3ad13dd92049c824c1d9`, but lock/create are still blocked by source transfer plus CoppeliaSim/PyRep external runtime setup. The branch now fails fast instead of entering the known-hanging resolver path. |
| LIBERO | `fallback_blocked` | `feat/sim-libero` (`108de40`) | Not merged | LIBERO is pinned to commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, but uv source resolution and runtime validation are blocked. Use a conda or container fallback until a Python 3.10 compatibility runtime passes required smoke. |

## Merge Rule

A backend should not move from the separate-branch table into the integrated table until these are true:

1. Its support level is explicit in `scripts/manage_uv_env.py`.
2. Any upstream source dependency is pinned to an exact commit and installed through the source checkout path.
3. `python3 scripts/manage_uv_env.py list` shows that support level.
4. `create` and `smoke` either pass on the target host or fail fast with a documented `blocked_reason`.
5. The backend imports remain lazy enough that `smoke core` passes without the simulator installed.
