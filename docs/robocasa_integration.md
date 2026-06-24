# RoboCasa Integration

RoboCasa / RoboCasa365 is the semantic household manipulation candidate backend. The minimal integration uses the official Gym wrapper style:

```python
gym.make("robocasa/PickPlaceCounterToCabinet", split="pretrain", seed=seed)
```

The first benchmark config is `configs/robocasa_pick_place_counter_to_cabinet.yaml`.

## Backend Boundary

- `simulator.type: robocasa` builds `RoboCasaSimAdapter`.
- `task.type: robocasa_generic` builds `RoboCasaGenericTask`.
- `controller.type: robocasa_osc` builds `RoboCasaOSCController`.
- Heavy imports stay inside `RoboCasaSimAdapter.reset`; core imports and config building do not import `robocasa`.

RoboCasa's Gym wrapper action space is a `Dict`, not the robosuite 7D OSC array. The controller emits:

- `action.end_effector_position`: 3D delta position.
- `action.end_effector_rotation`: 3D delta rotation.
- `action.gripper_close`: 1D close command in `[0, 1]`.
- `action.base_motion`: 4D zero vector for this minimal integration.
- `action.control_mode`: 1D mode value, default `1.0`.

The base-motion channels are intentionally fixed at zero for the first pass. Add mobile-base joystick dimensions only after validating the upstream controller semantics for the selected robot.

## Environment

Use the isolated uv backend:

```bash
python3 scripts/manage_uv_env.py create robocasa
python3 scripts/manage_uv_env.py smoke robocasa
```

Pinned dependencies in `envs/uv/robocasa.in`:

- RoboCasa commit `8f3c96ec8d1bfcd8126cad2bca887da98d30e997` (`v1.0` / RoboCasa365 release).
- robosuite `1.5.2` from PyPI. The earlier Git URL for the same release can hang in `uv` during Git fetch / metadata extraction.
- `numpy==2.2.5` and `mujoco==3.3.1`, matching RoboCasa v1.0 import assertions.

`scripts/manage_uv_env.py lock robocasa` uses the `x86_64-manylinux_2_31` target for this backend because `lerobot==0.3.3` requires `rerun-sdk>=0.21,<0.23`, and the compatible `rerun-sdk==0.22.1` wheel is published as `manylinux_2_31_x86_64`. It also uses the CPU Torch backend to avoid CUDA wheels for this CPU smoke.

`envs/uv/robocasa-excludes.in` excludes `mink` from resolution. robosuite `1.5.2` declares `mink==0.0.5`, whose `numpy<2` requirement conflicts with RoboCasa v1.0's `numpy==2.2.5`; RoboCasa's selected kitchen runtime does not use `mink` directly.

RoboCasa v1.0's built wheel omits `robocasa/scripts`, but `robocasa/utils/env_utils.py` imports `robocasa.scripts.dataset_scripts.playback_dataset_hdf5` at package import time. The create step therefore clones the same fixed RoboCasa commit into `.venvs/_sources/robocasa` and installs it editable with `--no-deps` after syncing the lock.

## Current Setup Status

As of 2026-06-24:

- `python3 scripts/manage_uv_env.py lock robocasa` completes and writes `envs/uv/locks/robocasa.lock.txt`.
- `python3 scripts/manage_uv_env.py create robocasa` completes, including the fixed RoboCasa editable source install.
- `python3 scripts/manage_uv_env.py smoke robocasa` exits successfully only because the runtime reset test is optional and skips. It is not runtime verification.

The remaining runtime blocker is robosuite source compatibility. PyPI `robosuite==1.5.2` lets uv resolve quickly, but its `ManipulationEnv.__init__` does not accept RoboCasa's `load_model_on_init=False` call. The matching robosuite source commit is still `824ac14cefcfb7ec125fe5eb2e0bad7364466154`; both uv Git resolution and a direct `git clone https://github.com/ARISE-Initiative/robosuite.git` exceeded 5 minutes in this workspace.

Next command sequence once GitHub clone/fetch is available:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git .venvs/_sources/robosuite
git -C .venvs/_sources/robosuite checkout 824ac14cefcfb7ec125fe5eb2e0bad7364466154
uv pip install --python .venvs/robocasa --no-deps -e .venvs/_sources/robosuite
MLLM_TELEOP_REQUIRE_ROBOCASA=1 python3 scripts/manage_uv_env.py smoke robocasa
```

Do not mark RoboCasa runtime verified until the required smoke runs without skipping.

Upstream references:

- RoboCasa repository: https://github.com/robocasa/robocasa
- RoboCasa v1.0 release: https://github.com/robocasa/robocasa/releases/tag/v1.0
- robosuite v1.5.2 tag: https://github.com/ARISE-Initiative/robosuite/releases/tag/v1.5.2

## Assets

RoboCasa assets are about 10 GB. They are not downloaded by tests or by importing this package.

After creating the uv environment, run:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python -m robocasa.scripts.setup_macros
python3 scripts/manage_uv_env.py run robocasa -- python -m robocasa.scripts.download_kitchen_assets
```

The upstream downloader fetches zip files from Hugging Face datasets and extracts them under the installed `robocasa` package:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python -c "import robocasa; print(robocasa.__path__[0] + '/models/assets')"
```

The runtime smoke test skips when the package, compatible robosuite source, or assets are missing. Set `MLLM_TELEOP_REQUIRE_ROBOCASA=1` to make those failures hard failures during backend validation.

## Validation

Run tests that do not require assets:

```bash
python3 -m pytest -q tests/test_robocasa_integration.py -rs
```

Run the actual backend smoke only after resolving the robosuite source blocker and installing assets:

```bash
python3 scripts/manage_uv_env.py smoke robocasa
python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml
```
