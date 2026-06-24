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
- robosuite commit `824ac14cefcfb7ec125fe5eb2e0bad7364466154` (`v1.5.2`).
- `numpy==2.2.5` and `mujoco==3.3.1`, matching RoboCasa v1.0 import assertions.

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

The runtime smoke test skips when the package or assets are missing. Set `MLLM_TELEOP_REQUIRE_ROBOCASA=1` to make those failures hard failures during backend validation.

## Validation

Run tests that do not require assets:

```bash
python3 -m pytest -q tests/test_robocasa_integration.py -rs
```

Run the actual backend smoke after installing dependencies and assets:

```bash
python3 scripts/manage_uv_env.py smoke robocasa
python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa_pick_place_counter_to_cabinet.yaml
```
