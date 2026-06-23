# Environment Management

This project uses conda as the only supported environment manager.

The repository is shared by multiple backend-specific conda environments. The root env installs this source tree with `-e .`; backend env files under `envs/` install it with `-e ..`, so code changes are immediately visible after switching environments.

## Policy

- Use one conda environment per simulator backend.
- Keep the root `environment.yml` lightweight and core-only.
- Do not install every simulator dependency into one environment.
- Do not require `uv` for setup, tests, or runtime.
- Use conda/micromamba to create and activate environments; the `pip:` section is only used inside conda env files to install Python packages and this repo in editable mode.

## Environments

Core development:

```bash
conda env create -f environment.yml
conda activate mllm-teleop-core
pytest
python scripts/run_benchmark.py --config configs/dummy_reach.yaml
```

Gymnasium/MuJoCo Reacher and Pusher:

```bash
conda env create -f envs/environment-gymnasium.yml
conda activate mllm-teleop-gymnasium
python scripts/run_benchmark.py --config configs/reacher_delta_eef.yaml
python scripts/play_benchmark.py --config configs/pusher_delta_eef.yaml
```

ManiSkill:

```bash
conda env create -f envs/environment-maniskill.yml
conda activate mllm-teleop-maniskill
python scripts/run_benchmark.py --config configs/maniskill_pickcube_delta_eef.yaml
```

robosuite:

```bash
conda env create -f envs/environment-robosuite.yml
conda activate mllm-teleop-robosuite
python scripts/run_benchmark.py --config configs/robosuite_lift.yaml
```

The robosuite environment pins `robosuite==1.4.1`. LIBERO should either reuse this environment only after compatibility is confirmed or receive its own backend environment file.

DMControl:

```bash
conda env create -f envs/environment-dmcontrol.yml
conda activate mllm-teleop-dmcontrol
python scripts/run_benchmark.py --config configs/dmcontrol_example.yaml
```

CALVIN:

```bash
conda env create -f envs/environment-calvin.yml
conda activate mllm-teleop-calvin
python scripts/run_benchmark.py --config configs/calvin_delta_eef.yaml
```

CALVIN is installed from the pinned upstream GitHub commit in editable mode because `calvin_env` is not published on PyPI and its runtime Hydra configs live outside the Python package wheel.

## Editable Install Check

After activating any backend environment, verify that the package points to this repository:

```bash
python -m pip show mllm-teleop-benchmark
```

The `Editable project location` should be the repository root.

## Adding A New Backend

When a new simulator backend is added:

1. Add `envs/environment-<backend>.yml`.
2. Keep the backend package imports lazy so core tests pass without the backend installed.
3. Add tests that skip or mock the heavy dependency when it is absent.
4. Document the environment in this file and the backend-specific integration doc.
5. Avoid adding backend dependencies to root `environment.yml` or core `pyproject.toml` dependencies.
