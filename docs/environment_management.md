# Environment Management

This project uses uv as the primary environment manager. Simulator backends still use separate environments so incompatible or heavy dependencies are not installed together.

Conda env files under `envs/environment-*.yml` remain as a fallback during the uv migration, but new setup and smoke-test commands should use `scripts/manage_uv_env.py`.

## Policy

- Use one uv virtual environment per simulator backend.
- Keep backend requirements in `envs/uv/*.in` and checked-in locks in `envs/uv/locks/*.lock.txt`.
- Do not install every simulator dependency into one environment.
- Do not commit `.venvs/` or CALVIN's upstream source checkout.
- Use CPU PyTorch for `maniskill-cpu` by default; use `maniskill-cu126` on NVIDIA CUDA 12.6 hosts.
- Keep backend imports lazy so the core environment works without optional simulators installed.

## uv Commands

List known backends:

```bash
python3 scripts/manage_uv_env.py list
```

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

robosuite:

```bash
python3 scripts/manage_uv_env.py create robosuite
python3 scripts/manage_uv_env.py run robosuite -- python scripts/run_benchmark.py --config configs/robosuite_lift.yaml
```

DMControl:

```bash
python3 scripts/manage_uv_env.py create dmcontrol
python3 scripts/manage_uv_env.py run dmcontrol -- python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

CALVIN:

```bash
python3 scripts/manage_uv_env.py create calvin
python3 scripts/manage_uv_env.py run calvin -- python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml
```

CALVIN is special: uv can resolve the pinned Git dependency, but it cannot install an editable Git URL directly. The create step clones `calvin_env` to `.venvs/_sources/calvin-env`, checks out commit `797142c588c21e76717268b7b430958dbd13bf48`, initializes submodules, and installs that local checkout editable so CALVIN's runtime Hydra `conf/` directory is available.

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

## Adding A New Backend

When a new simulator backend is added:

1. Add `envs/uv/<backend>.in`.
2. Add a backend entry in `scripts/manage_uv_env.py`.
3. Generate `envs/uv/locks/<backend>.lock.txt`.
4. Keep heavy imports lazy so core tests pass without the backend installed.
5. Add smoke tests that skip or mock the heavy dependency when it is absent.
6. Document the backend here and in its integration doc.
