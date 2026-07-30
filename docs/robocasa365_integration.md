# RoboCasa365 CuttingToolSelection Integration

## Scope

The repository's pinned RoboCasa commit
`8f3c96ec8d1bfcd8126cad2bca887da98d30e997` is RoboCasa v1.0, the upstream
RoboCasa365 release. The pinned source currently registers 65 atomic datasets
and 252 composite datasets, or 317 dataset entries total. This integration
therefore does not claim complete 365-task coverage or direct full-suite
comparability.

`CuttingToolSelection` is one official target task in the upstream
`composite_unseen` task set. It asks the policy to select the knife or peeler
appropriate for the episode's food and place that tool on the cutting board.
The dedicated entry evaluates that one task; it is not a separate simulator
backend.

The three configs are:

| Config | Use | Control condition |
| --- | --- | --- |
| `configs/robocasa365_cutting_tool_selection.yaml` | Shared Pygame human play | `fixed + identity` |
| `configs/robocasa365_cutting_tool_selection_mllm_reflection.yaml` | Chat Completions baseline | visible `fixed + identity` |
| `configs/robocasa365_cutting_tool_selection_mllm_reflection_perturbed.yaml` | Chat Completions hidden-control condition | hidden `permuted_sign_flipped + random_rotation_3d` |

All three use:

- `simulator.type: robocasa` and
  `env_id: robocasa/CuttingToolSelection`.
- The official `target` split and explicit object registries
  `[objaverse, lightwheel]`.
- Ten episodes and the upstream dataset horizon of 800 steps.
- Public RGB from `robot0_agentview_left`, `robot0_agentview_right`, and
  `robot0_eye_in_hand`.
- The public episode instruction from
  `annotation.human.task_description`. The static fallback is not used when an
  upstream annotation is available, and a missing annotation fails loudly.
- `save_video: false`. Video is enabled only for an intentional run with the
  runtime override `--set logging.save_video=true`.

Both MLLM conditions reject an environment that is already successful
immediately after reset. They use the Chat Completions API, reasoning effort
`low`, `action_with_reflection`, all action history, and three historical image
steps.

## Control Contract

The configs copy the normalized PandaOmron OSC contract from the existing
RoboCasa MLLM integration, not the legacy scaled human config. Six ternary
motion channels control `dx`, `dy`, `dz`, `droll`, `dpitch`, and `dyaw`; the
seventh channel is a binary gripper command where `0` is open and `1` is
closed.

Controller scales remain `1.0`. The normalized upstream endpoints represent
approximately `0.05 m` translation and `0.5 rad` rotation, so applying the
legacy `0.05` or `0.1` controller scales would scale the command twice. The
native RoboCasa action also contains a four-value `action.base_motion` fixed at
zero and `action.control_mode: [1.0]`.

The public prompt contains the dynamic task instruction, control semantics,
three RGB views, and the model's own permitted history. It does not contain
reward or success oracles, simulator state, object/end-effector poses, native
actions, the target split, object registry names, asset revisions, or sampled
hidden mappings and rotations.

## Asset Provenance

The configs record the two asset revisions under `simulator.params` as
audit-only metadata. Simulator parameters are not included in the model prompt.

- `scripts/download_robocasa_assets.py` downloads textures and Objaverse assets
  from `robocasa/robocasa-assets` at revision
  `1b92c3d02ca4354984fec961357db0bff7b32166`.
- `scripts/download_robocasa_lightwheel_assets.py` downloads Lightwheel
  fixtures and objects from
  `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` at revision
  `420a04af939c34873e6839a586b70844baf28aab`.

Run the pinned downloaders only inside the isolated RoboCasa environment:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python scripts/download_robocasa_assets.py --type all
python3 scripts/manage_uv_env.py run robocasa -- python scripts/download_robocasa_lightwheel_assets.py --type all
```

## Static Validation

The integration test is intentionally simulator-free and API-free:

```bash
python3 -m pytest -q tests/test_robocasa365_integration.py
```

It verifies config loading; environment and task metadata; exact public
channels, scales, gripper semantics, and fixed-zero base action; dynamic public
instruction projection; private and audit metadata exclusion; hidden mapping
non-disclosure; and that the MLLM pair differs only by its intended control
complexity and run identity.

## Runtime Gates

Runtime readiness remains unclaimed until all of these are completed on a host
with the pinned RoboCasa environment, required assets, and working display or
EGL support:

1. Create the backend and install both pinned asset sets.
2. Run
   `MLLM_TELEOP_REQUIRE_ROBOCASA=1 python3 scripts/manage_uv_env.py smoke robocasa`.
3. Start human play with
   `configs/robocasa365_cutting_tool_selection.yaml` and verify all three public
   cameras plus the dynamic annotation.
4. Run all ten seeds for each MLLM condition, reject initially successful
   resets, and verify native action bounds and task success reporting.
5. Enable video only for a deliberate validation run by adding
   `--set logging.save_video=true`.

Example commands to run after those prerequisites are met:

```bash
python3 scripts/manage_uv_env.py run robocasa -- python scripts/play_benchmark.py --config configs/robocasa365_cutting_tool_selection.yaml
python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa365_cutting_tool_selection_mllm_reflection.yaml --set logging.save_video=true
python3 scripts/manage_uv_env.py run robocasa -- python scripts/run_benchmark.py --config configs/robocasa365_cutting_tool_selection_mllm_reflection_perturbed.yaml --set logging.save_video=true
```

The upstream RoboCasa protocol randomly samples 50 scenarios for evaluation
rollouts and reports average binary success over those rollouts. These
checked-in configs use ten episodes, so their results are a repository-level
evaluation slice and must not be presented as directly comparable to upstream
50-scenario results.
