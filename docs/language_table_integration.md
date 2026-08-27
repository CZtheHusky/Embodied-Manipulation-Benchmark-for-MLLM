# Language-Table Integration

## Scope

This backend integrates the official Google Research Language-Table PyBullet
environment through the benchmark's shared `SimAdapter` path. The first checked-in
task is the official four-block `blocktoblock` reward condition. Its instruction is
sampled by the upstream environment at reset and passed to the human or MLLM agent as
the benchmark task instruction.

The pinned upstream source is:

- repository: `https://github.com/google-research/language-table.git`
- commit: `dab1ca4df68cab1727c1d8f26b48252ab2f3d473`

The upstream repository has no Python package metadata. The environment manager
therefore verifies the exact checkout under `.venvs/_sources/language-table` and adds
that directory to `PYTHONPATH`; it does not create an untracked wheel or edit the
upstream source.

## Environment

Create and verify the isolated runtime:

```bash
python3 scripts/manage_uv_env.py create language-table
python3 scripts/manage_uv_env.py smoke language-table
```

The lock contains only the environment runtime and this repository's test/UI
dependencies. It intentionally excludes the upstream CLIP, Scenic, JAX, Reverb,
TF-Agents, and dataset/training stack. TensorFlow remains in the runtime because an
upstream environment utility imports `tensorflow` for PyBullet state serialization,
even though ordinary reset/step/render does not use the dataset or training APIs.

## Controls And Observations

- Native action: planar end-effector target delta `(x, y)`.
- Benchmark action: two virtual joysticks mapped to `dx` and `dy`.
- Magnitude 1: `0.01 m` along one workspace axis.
- Native RGB: the official `rgb` source key, `320x180`, without rendered instruction
  text or evaluator debug overlays.
- Success: the upstream sparse reward terminates the episode when the instructed
  object reaches the target region.

An MLLM command larger than 1 is split by the existing agent-command execution layer
before it reaches PyBullet. These internal simulator steps do not consume additional
agent-command budget.

## Run

Random-agent benchmark:

```bash
python3 scripts/manage_uv_env.py run language-table -- \
  python scripts/run_benchmark.py --config configs/language_table_block2block.yaml
```

Human play:

```bash
python3 scripts/manage_uv_env.py run language-table -- \
  python scripts/play_benchmark.py --config configs/language_table_block2block.yaml
```

Chat Completions MLLM evaluation:

```bash
python3 scripts/manage_uv_env.py run language-table -- \
  python scripts/run_benchmark.py \
  --config configs/language_table_block2block_mllm_chat_completions.yaml
```

Responses MLLM evaluation with complete image and action history:

```bash
python3 scripts/manage_uv_env.py run language-table -- \
  python scripts/run_benchmark.py \
  --config configs/language_table_block2block_mllm_responses.yaml
```

The MLLM receives the sampled instruction and public RGB. Upstream block poses,
reward, success, target distance, seed, and simulator action remain evaluator data and
are not projected into the model request.
