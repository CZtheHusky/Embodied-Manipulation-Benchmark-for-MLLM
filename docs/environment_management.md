# Environment Management

This project uses a uv-first hybrid environment policy. Simulator backends still use separate environments so incompatible or heavy dependencies are not installed together, but not every simulator can be represented as a pure PyPI lock.

Conda env files under `envs/environment-*.yml` remain as a fallback during the uv migration. New setup and smoke-test commands should use `scripts/manage_uv_env.py` unless a backend is explicitly marked as blocked or fallback-only.

## Policy

- Use one isolated environment per simulator backend.
- Prefer uv locks in `envs/uv/*.in` and `envs/uv/locks/*.lock.txt` whenever the backend can be resolved and installed reproducibly.
- Do not install every simulator dependency into one environment.
- Do not commit `.venvs/` or upstream source checkouts under `.venvs/_sources/`.
- Classify every backend with one support level:
  - `uv`: uv lock plus normal package install.
  - `uv_source`: uv lock plus pinned source checkouts installed after sync.
  - `external_runtime`: uv-managed Python dependencies plus a required host runtime, simulator binary, or driver stack.
  - `fallback_blocked`: uv is not currently reliable enough; use a documented conda or container fallback instead.
- For source-backed environments, declare each upstream repository with a
  40-character commit SHA. Environment creation, `run`, and `smoke` require
  both the exact `HEAD` and no tracked modifications; untracked downloaded assets are allowed.
  Floating branches, tags, dirty tracked sources, and unpinned Git URLs are not
  valid runtime verification inputs.
- Do not mix uv and conda inside the same backend environment. If a backend uses a conda fallback, treat that conda env as the primary environment for that backend.
- Use CPU PyTorch for `maniskill-cpu` by default; use `maniskill-cu126` on NVIDIA CUDA 12.6 hosts.
- Keep backend imports lazy so the core environment works without optional simulators installed.
- Keep current backend status in `docs/simulator_backend_status.md`.
- Treat CoppeliaSim/PyRep as an external runtime prerequisite; uv can isolate Python packages but cannot install or validate the CoppeliaSim system binary.

## uv Commands

List known backends:

```bash
python3 scripts/manage_uv_env.py list
```

The list output is tab-separated: `<backend>\t<support_level>`.

For `uv_source` backends, `create` runs in this order:

1. Sync the uv lock.
2. Install this repository editable.
3. Fetch each declared upstream source into `.venvs/_sources/<name>`.
4. Check out and verify the exact commit.
5. Install that checkout into the target backend environment.

Regenerate a lock:

```bash
python3 scripts/manage_uv_env.py lock gymnasium
```

Create or refresh a backend environment. This clears `.venvs/<backend>` before syncing from the lock:

```bash
python3 scripts/manage_uv_env.py create gymnasium
```

Run a command inside a backend environment:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/run_benchmark.py --config configs/reacher_delta_eef.yaml
```

Run a backend smoke test:

```bash
python3 scripts/manage_uv_env.py smoke gymnasium
```

## Environments

Core development:

```bash
python3 scripts/manage_uv_env.py create core
python3 scripts/manage_uv_env.py smoke core
```

Gymnasium/MuJoCo Reacher and Pusher:

```bash
python3 scripts/manage_uv_env.py create gymnasium
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/run_benchmark.py --config configs/reacher_delta_eef.yaml
python3 scripts/manage_uv_env.py run gymnasium -- python scripts/play_benchmark.py --config configs/pusher_delta_eef.yaml
```

ManiSkill CPU:

```bash
python3 scripts/manage_uv_env.py create maniskill-cpu
python3 scripts/manage_uv_env.py run maniskill-cpu -- python scripts/run_benchmark.py --config configs/maniskill_pickcube_delta_eef.yaml
```

ManiSkill CUDA 12.6:

```bash
python3 scripts/manage_uv_env.py create maniskill-cu126
python3 scripts/manage_uv_env.py smoke maniskill-cu126
python3 scripts/manage_uv_env.py run maniskill-cu126 -- python scripts/run_benchmark.py --config configs/maniskill_pickcube_delta_eef.yaml
```

The `maniskill-cu126` smoke test checks `torch.cuda.is_available()` before running ManiSkill tests. A host without an NVIDIA driver should fail this smoke test instead of pretending the GPU environment is valid.

SimplerEnv:

```bash
python3 scripts/manage_uv_env.py create simpler-env
python3 scripts/manage_uv_env.py smoke simpler-env
python3 scripts/manage_uv_env.py run simpler-env -- python scripts/run_benchmark.py --config configs/simpler_env_google_robot_pick_coke_can.yaml
```

The `simpler-env` backend installs the minimal environment runtime only: `simpler_env`, `mani_skill2_real2sim`, and their simulator dependencies. It intentionally does not install RT-1, Octo, TensorFlow/JAX policy inference, or the full SimplerEnv development stack.

The lock file contains the PyPI runtime dependencies. `create simpler-env` then fetches pinned upstream source checkouts declared in `scripts/manage_uv_env.py` into `.venvs/_sources/` and installs them editable with `--no-deps`:

- `mani_skill2_real2sim`: `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`
- `simpler_env`: `06accaca93535902d408da4855f21cece12bceb7`

This avoids uv resolver hangs on upstream Git URL dependencies while keeping the source revisions explicit. `sapien==2.2.2` still imports `pkg_resources`, so the backend pins `setuptools<81`.

SimplerEnv is SAPIEN based and expects a Linux NVIDIA GPU setup for practical rendering; missing GPU drivers, incompatible CUDA/SAPIEN wheels, or missing upstream assets should be treated as backend setup failures rather than core benchmark failures. Once `simpler_env` is installed, the smoke test fails on import/reset/render/step errors instead of skipping them.

robosuite:

```bash
python3 scripts/manage_uv_env.py create robosuite
python3 scripts/manage_uv_env.py run robosuite -- python scripts/run_benchmark.py --config configs/robosuite_lift.yaml
```

RoboCasa / RoboCasa365:

```bash
python3 scripts/manage_uv_env.py create robocasa
python3 scripts/manage_uv_env.py run robocasa -- python -m robocasa.scripts.setup_macros
python3 scripts/manage_uv_env.py run robocasa -- python scripts/download_robocasa_lightwheel_assets.py --type all
python3 scripts/manage_uv_env.py run robocasa -- python -c "import builtins; builtins.input=lambda prompt='': 'y'; from robocasa.scripts.download_kitchen_assets import download_kitchen_assets; download_kitchen_assets(['tex', 'tex_generative'])"
python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml
```

RoboCasa assets are around 10 GB and are downloaded from Hugging Face datasets by the upstream script. The asset folders are created under the installed `robocasa` package, for example `$(python -c "import robocasa; print(robocasa.__path__[0])")/models/assets`. Do not run the asset download from ordinary unit tests; `tests/test_robocasa_integration.py` skips the runtime smoke when RoboCasa or its assets are unavailable.

RoboCasa v1.0 needs a backend-specific uv setup: `lock robocasa` targets `x86_64-manylinux_2_31`, uses CPU Torch wheels, and excludes robosuite's `mink` dependency because it conflicts with RoboCasa's `numpy==2.2.5`. The create step also installs the fixed RoboCasa source checkout editable from `.venvs/_sources/robocasa`, because the built wheel omits `robocasa/scripts` required during `import robocasa`.

Current RoboCasa state: lock, create, required asset download, runtime smoke, and one random-agent benchmark episode are verified. The previous robosuite `load_model_on_init` API blocker is resolved by pinning the upstream `master` instruction to robosuite commit `85abee228d1c43ab1939bce33028099945d453b4`. RoboCasa v1.0 expects monolithic Lightwheel asset zips, but the current NVIDIA Hugging Face dataset publishes split zip files; `scripts/download_robocasa_lightwheel_assets.py` pins dataset revision `420a04af939c34873e6839a586b70844baf28aab` and restores the expected layout. For large texture downloads behind Clash, the `robocasa` uv runner disables Hugging Face Xet and removes `ALL_PROXY/all_proxy` so downloads keep using the HTTP proxy path instead of the SOCKS/Xet path.

The same runner also enforces `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl`, so non-interactive benchmark commands receive the
offscreen rendering environment instead of relying on the Pygame UI hook.

DMControl:

```bash
python3 scripts/manage_uv_env.py create dmcontrol
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

Meta-World MT1 reach-v3:

```bash
python3 scripts/manage_uv_env.py create metaworld
python3 scripts/manage_uv_env.py smoke metaworld
python3 scripts/manage_uv_env.py run metaworld -- python scripts/run_benchmark.py --config configs/metaworld_reach.yaml
```

CALVIN:

```bash
python3 scripts/manage_uv_env.py create calvin
python3 scripts/manage_uv_env.py run calvin -- python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml
```

CALVIN is special: uv can resolve the pinned Git dependency, but it cannot install an editable Git URL directly. The create step fetches `calvin_env` to `.venvs/_sources/calvin-env`, verifies commit `797142c588c21e76717268b7b430958dbd13bf48`, initializes submodules, and installs that local checkout editable so CALVIN's runtime Hydra `conf/` directory is available.

RLBench:

```bash
python3 scripts/manage_conda_env.py install-coppeliasim
python3 scripts/manage_conda_env.py create rlbench
python3 scripts/manage_conda_env.py smoke rlbench
```

RLBench is runtime verified through the Miniconda fallback in this branch. `scripts/manage_conda_env.py` installs the external CoppeliaSim v4.1.0 binary under `/home/caozhe/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`, creates `mllm-teleop-rlbench`, and installs pinned source commits for PyRep and RLBench. The uv route is not the supported route for this backend because the lock path stalled on the pinned Git source and still would not manage the external CoppeliaSim binary.

The helper sets `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH`, and uses `DISPLAY=:1` when that X socket exists. The verified path is display-backed OpenGL; headless GPU validation is not claimed on this host.

Colosseum uses its own Miniconda backend:

```bash
python3 scripts/manage_conda_env.py create colosseum
python3 scripts/manage_conda_env.py smoke colosseum
```

This environment uses the official Colosseum commit
`098a910ae1d490524adb49b8d50735b00a1a62de`, RLBench commit
`7c3f425f4a0b6b5ce001ba7246354eb3c70555be`, PyRep commit
`076ca15c57f2495a4194da03565891ab1aaa317e`, and CoppeliaSim 4.1. Its source
checkout names are separate from the `rlbench` backend, so creating Colosseum
does not change the existing RLBench pin. The helper applies the same
CoppeliaSim environment variables and display fallback. Use
`MLLM_TELEOP_REQUIRE_COLOSSEUM=1` to turn an unavailable external runtime into a
test failure.

LIBERO:

```bash
python3 scripts/manage_conda_env.py create libero
python3 scripts/manage_conda_env.py smoke libero
```

LIBERO is runtime verified through the Miniconda fallback in this branch. The helper pins upstream LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, installs the minimal runtime stack needed for this benchmark adapter, writes non-interactive LIBERO and robosuite config files, and removes `ALL_PROXY/all_proxy` so pip keeps using the local HTTP/HTTPS Clash proxy instead of the SOCKS path.

The uv commands intentionally fail fast for LIBERO because that path is not runtime verified. The conda route is a benchmark runtime subset, not a full reproduction of LIBERO's training and policy stack. Use `docs/libero_integration.md` for details.

## Editable Install Check

After creating any backend environment, verify that the benchmark package points to this repository:

```bash
python3 scripts/manage_uv_env.py run gymnasium -- python -c "import mllm_teleop_benchmark; print(mllm_teleop_benchmark.__file__)"
```

The printed path should be under this repository's `src/` directory.

## Conda Fallback

Existing conda files remain available for rollback while uv environments are being validated:

```bash
conda env create -f envs/environment-gymnasium.yml
conda activate mllm-teleop-gymnasium
```

Do not add new simulator dependencies to the root `environment.yml`. New backends should add uv input and lock files first, then add a conda fallback file only when there is a concrete reason.

A fallback backend should fail fast from `scripts/manage_uv_env.py` with a clear reason and a documentation link. Do not run `uv pip sync` inside a conda fallback environment; that produces an untracked hybrid state that cannot be recreated from either manager alone.

## Adding A New Backend

When a new simulator backend is added:

1. Add `envs/uv/<backend>.in`.
2. Classify the backend as `uv`, `uv_source`, `external_runtime`, or `fallback_blocked`.
3. Add a backend entry in `scripts/manage_uv_env.py`. A blocked backend must include `blocked_reason` and `doc`; a source backend must include exact source commits in `sources`.
4. Generate `envs/uv/locks/<backend>.lock.txt` unless the backend is `fallback_blocked`.
5. Keep heavy imports lazy so core tests pass without the backend installed.
6. Add smoke tests that skip or mock the heavy dependency when it is absent.
7. Document the backend here, in its integration doc, and in `docs/simulator_backend_status.md`.
8. Run `python3 scripts/manage_uv_env.py list` and a backend smoke test before merging.
