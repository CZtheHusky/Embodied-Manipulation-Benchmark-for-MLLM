# LIBERO Integration

LIBERO is added as an optional language-conditioned manipulation backend. The goal is a minimal reproducible adapter for evaluating closed-loop visual feedback, not a claim that standard LIBERO scores are robust language-understanding measurements.

## Status

Initial config:

- `configs/libero_spatial_task0.yaml`
- `configs/libero_spatial_task0_mllm_reflection.yaml`
- `configs/libero_spatial_task0_mllm_reflection_perturbed.yaml`
- `simulator.type: libero`
- `task.type: libero_generic`
- `controller.type: libero_delta_eef`
- Suite/task/init state: `libero_spatial`, task `0`, init-state `0`

The official API checked for this adapter is:

```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task = task_suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=task_suite.get_task_bddl_file_path(0))
env.reset()
env.set_init_state(task_suite.get_task_init_states(0)[0])
obs, reward, done, info = env.step([0.0] * 7)
```

This matches the upstream README and current `env_wrapper.py`: `OffScreenRenderEnv` wraps robosuite and exposes `reset`, `set_init_state`, `step`, `check_success`, `sim`, and `close`.

Dependency status on 2026-07-29: LIBERO is runtime verified through the Miniconda fallback in this branch. The uv setup remains blocked by the old upstream dependency stack and is not the supported route for this backend.

Verified commands:

```bash
python3 scripts/manage_conda_env.py create libero
python3 scripts/manage_conda_env.py smoke libero
```

The smoke result was:

- `tests/test_libero_integration.py`: `15 passed`, no skips
- `configs/libero_spatial_task0.yaml`: one random-agent benchmark episode, `average_steps=200.0`, `success_rate=0.0`
- `scripts/play_benchmark.py --config configs/libero_spatial_task0.yaml --set ui.max_steps=0`: shared Pygame worker runtime starts LIBERO, extracts the `agentview` frame, and exits cleanly.

The benchmark success rate is expected to be zero for the random policy. The verification claim is reset, init-state application, one environment step, image/render access, close, and benchmark-runner completion.

## Architecture

The integration follows the project boundaries:

- `LiberoSimAdapter`: owns LIBERO/robosuite environment creation, reset, set-init-state, step, render, state extraction, and close.
- `LiberoDeltaEEFController`: owns joystick mapping, sign, scale, frame transform, and final 7D action creation.
- `LiberoObservationProvider`: extracts `*_image` RGB observations before falling back to `sim.render`.
- `LiberoTask`: provides instruction, success, metrics, and benchmark-validity metadata.

Heavy imports are lazy. Importing core registry, config loading, controllers, and tests does not import `libero`, `robosuite`, or `torch`.

LIBERO exposes robosuite's raw OpenGL camera rows in bottom-up order. The
adapter flips each observation image exactly once before it becomes public RGB;
the direct `sim.render()` fallback applies the same normalization. Non-symmetric
pixel canaries cover both paths so model-visible frames cannot silently become
vertically inverted.

For the resulting upright `agentview`, image right is approximately `+world y`
(`[0, 1, 0]`) and image up is approximately
`-0.628 world x + 0.778 world z` (`[-0.628, 0, 0.778]`). The MLLM configs store
these approximate MuJoCo-world basis vectors, and lightweight tests verify their
components, norms, and orthogonality.

## Control

The action path is:

```text
JoystickAction -> JoystickMapping -> ControlFrameTransform -> LiberoDeltaEEFController -> LiberoAction -> OffScreenRenderEnv.step()
```

The native action order is fixed:

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

The legacy human-play config applies benchmark-side scales of `0.05` for translation and `0.1` for rotation. The MLLM configs instead send normalized `[-1, 1]` pose commands because robosuite's `OSC_POSE` controller performs its own scaling from normalized input to physical deltas: translation output is bounded at `0.05 m` and the axis-angle rotation-vector output at `0.5 rad` per controller command. Applying both scales would make MLLM motion unintentionally small.

The six pose deltas use the MuJoCo world Cartesian frame. This follows
robosuite `OSC_POSE`: translation deltas are added to the current world-space
end-effector site position, and rotation is a right-hand axis-angle rotation
vector applied in the world frame. There is no backend sign compensation.
Positive normalized components remain positive world-frame components. The
gripper virtual command is binary: `0` means open and maps to native `-1`; `1`
means closed and maps to native `1`.

The perturbed MLLM condition applies `permuted_sign_flipped` to the six motion dimensions and one episode-fixed `random_rotation_3d` to both the translation and axis-angle rotation vectors. The gripper remains direct. Rotated motion vectors are rescaled only when needed to preserve their direction while staying within LIBERO's normalized native action range.

## MLLM Evaluation

The two switchable MLLM conditions use identical upstream language text:

```text
pick up the black bowl between the plate and the ramekin and place it on the plate
```

Both use:

- `action_with_reflection`
- OpenAI-compatible Chat Completions
- reasoning effort `low`
- three historical image turns and all textual action history
- public `agentview` and `robot0_eye_in_hand` RGB
- one fixed upstream `libero_spatial` task with deterministic seed-selected
  official init states

The baseline exposes the fixed joystick mapping and identity frame. The perturbed condition tells the model that the mapping and frame transform are hidden and episode-fixed, but never includes the sampled permutation, signs, rotation matrix, quaternion, simulator state, BDDL goal predicates, reward, success oracle, native action, or seed in the request.

The human-play config keeps explicit `init_state_id: 0`. The MLLM configs use
`init_state_strategy: seed_modulo`, selecting `episode_seed % init_state_count`.
This makes repeated runs deterministic while allowing a ten-episode batch to
cover distinct official initial states. The selected id is evaluator metadata
and is never included in the model request.

The local Fake Client validation runs four real simulator steps for each condition. It checks:

- both public camera streams are non-black `256x256` RGB
- each translated native action has shape `(7,)`, finite values, and satisfies the upstream robosuite `action_spec`
- positive normalized `dx`, `dy`, and `dz` commands move the measured end-effector position along the corresponding MuJoCo world axis
- `LiberoSimAdapter._check_success()`, `OffScreenRenderEnv.check_success()`, and the underlying BDDL environment `_check_success()` agree
- the parsed BDDL goal state is non-empty and fixed during the episode
- reflection/action history and the three-image-turn policy are preserved
- poisoned privileged-state and API-key sentinels never enter model requests or persisted artifacts

This establishes runtime and oracle wiring, not task completion by a real model. No real API request is part of the integration test.

## Benchmark Validity

Every reset records:

- language instruction
- suite name
- task id
- upstream task name
- init-state id
- init-state selection strategy
- `no_language_baseline`
- `random_baseline`

These fields are included in observation metadata, simulator state, and task metrics under `benchmark_validity`.

The default config uses the random agent and marks `random_baseline: true`. For a shortcut baseline that removes language, run with:

```bash
python3 scripts/run_benchmark.py --config configs/libero_spatial_task0.yaml --set task.params.no_language_baseline=true
```

Do not report LIBERO success without also reporting these metadata fields and the baseline setting. Recent robustness work has shown that standard LIBERO evaluation can reward memorized action/layout shortcuts, so LIBERO should be treated as a candidate benchmark until no-language, random, and perturbation baselines are compared.

## Dependencies

The upstream official conda reproduction path is:

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

That official Python 3.8.13 environment is useful for validating upstream LIBERO itself, but it is not the runtime for this repository because `pyproject.toml` requires Python >=3.10. Do not weaken the repository-wide Python requirement to fit LIBERO.

This repository uses a separate Miniconda compatibility environment that keeps Python 3.10 and installs only the runtime subset needed for `OffScreenRenderEnv`, benchmark metadata, and the benchmark runner:

```bash
python3 scripts/manage_conda_env.py create libero
python3 scripts/manage_conda_env.py smoke libero
```

Manual play uses the same conda backend:

```bash
python3 scripts/manage_conda_env.py run libero -- python scripts/play_benchmark.py --config configs/libero_spatial_task0.yaml
```

The script pins upstream LIBERO to commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` and writes a `.pth` file because that repository's root `setup.py` installs metadata but does not expose the nested `libero.libero` package correctly in this Python 3.10 editable install.

The verified compatibility stack is:

- Torch `1.11.0+cpu`
- robosuite `1.4.0`
- MuJoCo `2.3.7`
- NumPy `1.22.4`
- SciPy `1.10.1`
- numba `0.57.1`
- gym `0.25.2`
- bddl `1.0.1`
- matplotlib `3.5.3`

This is deliberately not the full LIBERO training stack. `wandb`, `transformers`, `robomimic`, `torchvision`, and `torchaudio` are not needed for the runtime adapter smoke and should not be pulled into this environment unless a policy-training/evaluation workflow is added and verified separately.

Proxy note: pip downloads used `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897`. The conda helper removes `ALL_PROXY/all_proxy` so pip does not accidentally take a SOCKS path through Clash.

## Smoke Test

The runtime smoke in `tests/test_libero_integration.py` skips when LIBERO is not required and LIBERO, robosuite, BDDL files, init states, assets, or headless rendering are unavailable. Set `MLLM_TELEOP_REQUIRE_LIBERO=1` in a claimed LIBERO runtime environment so missing dependencies or runtime failures fail loudly. When the stack is installed, it verifies:

- config and controller build
- reset
- `set_init_state`
- one legacy 7D action step
- four real MLLM Fake Client steps for each control condition
- normalized native action validation
- two-camera image extraction and non-black RGB
- BDDL success-oracle agreement
- public prompt isolation
- close
