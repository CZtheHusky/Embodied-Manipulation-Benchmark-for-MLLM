# Potential Benchmark and Simulation Environment Survey

Date: 2026-06-23

This report surveys candidate manipulation benchmarks and virtual environments that could extend the current `Embodied-Manipulation-Benchmark-for-MLLM` project. The goal is not to find the most popular robotics benchmark in isolation, but to identify environments that can fit this repository's evaluation model: closed-loop visual feedback, a structured virtual joystick interface, benchmark-layer control perturbations, reproducible logging, and backend-specific `uv` environments.

## Project Fit Criteria

The current project design makes several constraints non-negotiable:

- The agent-visible action interface is a virtual joystick schema over motion channels such as `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`, and optional final gripper control. See [benchmark_design.md](benchmark_design.md).
- Mapping permutation, sign flips, and control-frame rotation must stay in this benchmark's controller layer before the simulator receives the final action. See [architecture.md](architecture.md).
- New simulator integrations should implement `reset`, `step`, `render`, and `get_state`, keep heavy imports lazy, and use backend-specific `uv` lock files. See [architecture.md](architecture.md) and [environment_management.md](environment_management.md).
- Visual randomization should be done through runtime APIs during reset, not by mutating static simulator assets. See [architecture.md](architecture.md).
- A useful candidate needs reliable task success signals, reproducible seeds, visual observations, and a path to a narrow smoke test.

Evaluation axes used below:

- **Control fit**: whether actions can be mapped from benchmark joystick deltas into the environment.
- **Vision fit**: RGB/multi-view/camera support and whether camera perturbations are practical.
- **Task fit**: manipulation relevance, language/MLLM relevance, and task diversity.
- **uv fit**: feasibility of isolating dependencies under one backend-specific `uv` environment.
- **Benchmark risk**: risks such as shortcut solvability, stale dependencies, hard-to-randomize assets, unstable success metrics, or weak statistical signal.

## Executive Recommendation

Recommended order for real integration spikes:

1. **Meta-World**: best low-risk control and adapter spike. It uses Gymnasium-style APIs, has 50 manipulation tasks, and should be the fastest way to add broad manipulation coverage. It is weak on language and rich vision, so treat it as a controller/action robustness backend, not the main MLLM benchmark.
2. **RoboCasa / RoboCasa365**: best semantic household-manipulation candidate. It is built around everyday kitchen tasks, has a Gym wrapper, and depends on robosuite, which this repository already supports conceptually. It is heavier because of assets, robosuite master-branch expectations, and large datasets, but it is the strongest next benchmark if the objective is MLLM-style household manipulation.
3. **SimplerEnv**: best real-to-sim/VLA evaluation candidate. It exposes a 7D delta action style that is close to this project's full joystick interface and uses Gym-style interaction. It has fewer tasks than RoboCasa, but it is valuable for validating whether simulator evaluation correlates with real robot policy behavior.
4. **RLBench, then The Colosseum**: strong manipulation and visual generalization candidates if CoppeliaSim/PyRep can be tolerated. RLBench gives many tasks and action modes; Colosseum adds systematic perturbations. The install/headless-rendering burden is the main blocker.
5. **LIBERO**: valuable for language-conditioned/lifelong manipulation, but dependency age and benchmark-validity concerns make it less attractive than RoboCasa or SimplerEnv for immediate work.
6. **Isaac Lab / ARNOLD / BEHAVIOR-1K / OmniGibson**: high-fidelity long-term targets. They are important, but they should not be the next small integration because dependency, GPU, asset, and rendering complexity are high.

The largest benchmark-validity lesson from recent work is that raw success rate on a fixed setup is not enough. A 2026 audit identifies shortcut solvability, insufficient statistics, overfitting, and data-source dependence as common failure modes, and specifically reports issues in LIBERO and CALVIN while finding fewer issues in RoboCasa and RoboTwin 2.0. This should influence our own integration gates and reporting format. Source: [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233).

## Candidate Matrix

| Candidate | Fit | Why it matters | Main integration risk | Recommended status |
| --- | --- | --- | --- | --- |
| Meta-World | High for control, medium for MLLM | 50 manipulation tasks, Gymnasium API | Limited language/scene semantics and visual richness | Integrate first as a lightweight broad-task backend |
| RoboCasa / RoboCasa365 | High | Everyday kitchen manipulation, large task/asset scale, Gym wrapper, robosuite lineage | Large assets, robosuite dependency pinning, possible install drift | Prioritize after a focused uv/import spike |
| SimplerEnv | High for VLA/real-to-sim | 7D delta action, Gym API, real/sim correlation focus | SAPIEN/ManiSkill2 submodule, CUDA/GPU requirements, fewer tasks | Strong GPU-machine spike candidate |
| RLBench | High | Broad manipulation tasks, visual observations, action modes, demonstrations | CoppeliaSim/PyRep, headless X/GPU setup, external binary | Add only after CoppeliaSim install spike passes |
| The Colosseum | Medium-high | Perturbation-oriented manipulation benchmark | Likely inherits RLBench/CoppeliaSim complexity | Use after RLBench, mainly for robustness evaluation |
| LIBERO | Medium-high | Language-conditioned/lifelong robot learning, 130 tasks, 7D actions | Python/Torch stack is old; benchmark audit concerns | Keep as research target, not first integration |
| VLABench | Medium-high | Designed for VLA-style language-conditioned long-horizon tasks | Newer benchmark, unknown install/runtime maturity | Prototype only after code/assets are locally inspected |
| GemBench | Medium | Generalization benchmark with long-horizon language tasks | More benchmark/data oriented; action interface needs local verification | Watch/spike later |
| REALM | Medium | Real-to-sim validated VLA generalization benchmark with perturbations | Newer and likely high-fidelity/heavy; public maturity needs verification | Track closely, spike when code path is stable |
| RoboTwin 2.0 | Medium | Bimanual tasks, domain randomization, MLLM-assisted task generation | Current benchmark interface is single-arm; dual-arm needs schema extension | Long-term target after multi-arm action interface |
| RoboVerse / MetaSim | Medium | Simulator-agnostic abstraction and unified benchmark direction | Very broad/new; may duplicate this repo's adapter boundary | Track for interface ideas, not immediate backend |
| Isaac Lab | Medium | Modern GPU-accelerated robot learning framework | Heavy Isaac Sim dependency, AppLauncher/runtime model, GPU/container expectations | Long-term high-fidelity platform |
| ARNOLD | Medium | Isaac-based language-grounded tasks with continuous object states | Isaac dependency and narrow task count | Use as Isaac-specific benchmark reference |
| BEHAVIOR-1K / OmniGibson | Medium | Rich household activities and long-horizon tasks | Very heavy simulator/assets; action abstraction likely not joystick-native | Long-term household benchmark, not next adapter |
| AI2-THOR / ManipulaTHOR | Medium-low | Unity-based visual manipulation and mobile manipulation | Action API is less aligned with low-level EEF joystick control | Only if navigation+manipulation becomes central |
| Habitat Lab | Low-medium | Modular embodied AI platform | More navigation/rearrangement oriented than arm-control benchmark | Not near-term |
| VIMA-Bench | Medium for MLLM prompts, low-medium for joystick | Multimodal prompting benchmark for tabletop manipulation | Often closer to primitive/action-token benchmark than low-level teleop | Use for prompt/evaluation ideas |
| Ravens / CLIPort | Low-medium | Simple tabletop pick-place and language-conditioned manipulation baselines | Ravens is archived; primitive pick/place actions are not joystick-native | Do not prioritize as a new simulator backend |
| Raw SAPIEN / PartNet-Mobility | Low-medium | Strong simulator foundation for articulated objects | Better consumed through ManiSkill/SimplerEnv than directly | Avoid raw integration unless a specific task demands it |

## Detailed Findings

### Meta-World

Official sources: [project page](https://meta-world.github.io/) and [Farama GitHub](https://github.com/Farama-Foundation/Metaworld).

Meta-World provides 50 robotic manipulation environments for multi-task and meta-RL. The current maintained version follows Gymnasium-style creation and stepping, including MT1/MT10/MT50 and ML1/ML10/ML45 benchmark modes. That makes it the cleanest candidate for this repository's existing `GymnasiumSimAdapter` pattern.

Fit:

- Control: high. Continuous Box actions can be wrapped by a benchmark controller.
- Vision: medium. Rendering exists, but this is not its main strength compared with RLBench/RoboCasa.
- Task: medium. Good manipulation breadth, weak natural-language and household semantics.
- uv: high. Package installation is straightforward relative to CoppeliaSim or Isaac stacks.

Recommendation:

- Use the current `metaworld` MT1 backend as the quick task-diversity foothold.
- Extend beyond MT1 `reach-v3` only after action/state/success fields are stable across another single-task config.
- Treat it as a control and generalization sanity backend, not the headline MLLM benchmark.

### RoboCasa / RoboCasa365

Official sources: [RoboCasa project page](https://robocasa.ai/) and [GitHub](https://github.com/robocasa/robocasa).

RoboCasa targets everyday kitchen tasks and RoboCasa365 expands this to 365 tasks across 2,500 kitchen environments, with large human and synthetic demonstration datasets. The GitHub documentation shows a Gymnasium wrapper and explicitly depends on robosuite. This is strategically attractive because the repository already has robosuite integration concepts and controllers.

Fit:

- Control: high, pending action-space verification. It should be close to robosuite/OSC-style control.
- Vision: high. It is designed around diverse realistic kitchen scenes and textures.
- Task: high. Household manipulation and natural task semantics are aligned with MLLM evaluation.
- uv: medium. The official setup uses conda and a robosuite source checkout; assets are around 10GB.

Risks:

- Official instructions say to use the robosuite master branch. For this project, that must become a pinned commit or a local lock will not be reproducible.
- Asset download size and license/dataset terms must be documented before CI or smoke tests.
- A 2026 benchmark audit suggests RoboCasa has fewer validity failures than LIBERO/CALVIN, but we still need our own perturbation and confidence-interval reporting.

Recommendation:

- Run an isolated uv spike with only import, Gym env creation, `reset`, one sampled `step`, and one RGB render.
- If the spike passes, create a dedicated `RoboCasaSimAdapter` rather than overloading the existing robosuite adapter too early.
- First target: a single short-horizon pick/place kitchen task.

### SimplerEnv

Official sources: [project page](https://simpler-env.github.io/) and [GitHub](https://github.com/simpler-env/SimplerEnv).

SimplerEnv is explicitly about evaluating real-world robot manipulation policies in simulation. It exposes a uniform Gym API and its README shows an action layout of delta xyz, delta rotation, and gripper, which maps closely to this project's full joystick interface. It also includes visual matching and variant aggregation evaluation setups.

Fit:

- Control: high. The 7-value action style is a direct match for `dx/dy/dz/droll/dpitch/dyaw/gripper`.
- Vision: high for real-to-sim evaluation, but not necessarily broad multi-view benchmark design.
- Task: medium. It has practical Google Robot and WidowX task setups, but fewer tasks than RoboCasa/RLBench.
- uv: medium-low. The repo has submodules, SAPIEN/ManiSkill2 lineage, and CUDA/GPU expectations.

Recommendation:

- Use the GPU station for a smoke spike after confirming the existing ManiSkill CUDA setup remains valid.
- Treat it as a high-value evaluation backend for VLA-style policies and real-to-sim robustness, not as the only broad manipulation benchmark.

### RLBench

Official sources: [GitHub](https://github.com/stepjam/RLBench) and [paper](https://arxiv.org/abs/1909.12271).

RLBench is a large robot learning benchmark centered on vision-guided manipulation. The paper describes 100 hand-designed tasks with proprioceptive and visual observations, including external and eye-in-hand camera views. The GitHub README shows several action modes and examples using `MoveArmThenGripper` with joint velocity plus discrete gripper. It is built around CoppeliaSim and PyRep.

Fit:

- Control: high. RLBench action modes can likely support a benchmark controller, but the chosen action mode must be pinned.
- Vision: high. Multi-camera observations are a major strength.
- Task: high. Many manipulation tasks and demonstrations.
- uv: medium-low. CoppeliaSim is an external binary and headless rendering setup is non-trivial.

Recommendation:

- Do a CoppeliaSim/PyRep install spike before writing adapter code.
- If successful, implement only one task such as ReachTarget or OpenDoor with a fixed action mode.
- Keep CoppeliaSim as an external system prerequisite documented outside the uv lock.

### The Colosseum

Official source: [project page](https://robot-colosseum.github.io/).

The Colosseum focuses on generalization under perturbations across 20 manipulation tasks and 14 axes including color, texture, size, lighting, distractors, physical properties, and camera pose. This is unusually aligned with this repository's planned camera/control-complexity benchmarking.

Fit:

- Control: medium-high, likely dependent on RLBench-style integration details.
- Vision: high for perturbation testing.
- Task: high for robustness, medium for broad household semantics.
- uv: medium-low due to likely RLBench/CoppeliaSim complexity.

Recommendation:

- Do not integrate before RLBench.
- Once RLBench works, use Colosseum as a robustness suite rather than a separate first backend.

### LIBERO

Official sources: [project page](https://libero-project.github.io/) and [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO).

LIBERO provides 130 tasks grouped into task suites for spatial, object, goal, and LIBERO-100 evaluations. Its examples use language descriptions, offscreen rendering, fixed initial states, and 7-value dummy actions.

Fit:

- Control: high on shape, pending semantic mapping.
- Vision: high enough for image-based policy evaluation.
- Task: high for language-conditioned manipulation and lifelong transfer.
- uv: low-medium. Official setup is Python 3.8, older CUDA/Torch versions, and conda-oriented.

Risks:

- The 2026 benchmark audit reports LIBERO failing multiple diagnostics and says a small probe without a language encoder can score at or near reported SOTA. That does not make LIBERO useless, but it means success-rate claims need extra perturbation and non-language baselines.

Recommendation:

- Keep LIBERO as a useful comparison target, but do not make it the next integration unless the research question specifically targets lifelong language-conditioned imitation.
- If integrated, add shortcut checks from day one: no-language baseline, pose randomization beyond fixed initial states, camera perturbations, and confidence intervals.

### VLABench

Official source: [project page](https://vlabench.github.io/).

VLABench is a language-conditioned manipulation benchmark aimed at VLA models. It includes 100 task categories, more than 2,000 objects, primitive and composite tasks, implicit natural-language goals, and both interactive and non-interactive evaluation modes.

Fit:

- Control: unknown until code inspection.
- Vision/task: high for MLLM/VLA evaluation.
- uv: unknown-medium; likely manageable only after checking package and simulator dependencies.

Recommendation:

- Track as a strong MLLM/VLA candidate.
- Run a local code/install spike before committing to adapter work, because the public project page is stronger evidence for benchmark design than for operational integration.

### GemBench

Official source: [project page](https://www.di.ens.fr/willow/research/gembench/).

GemBench evaluates generalization of vision-language robotic manipulation policies with seven action primitives and four generalization levels across placements, rigid/articulated objects, and long-horizon tasks.

Fit:

- Task: high for generalization research.
- Control: medium-low until action API is inspected. The primitive structure may not map cleanly to this repository's joystick interface.
- uv: unknown.

Recommendation:

- Treat as a benchmark-design reference and later integration candidate, not a first backend.

### REALM

Official sources: [project page](https://martin-sedlacek.com/realm/) and [paper](https://arxiv.org/abs/2512.19562).

REALM is a real-to-sim validated VLA benchmark for generalization. It reports 7 manipulation skills, 15 perturbations, thousands of objects, high-fidelity visuals, aligned robot control, and strong sim-real correlation evidence.

Fit:

- Task/vision: high.
- Benchmark validity: high as a design direction because it explicitly validates sim-real correlation and perturbations.
- Control/uv: unknown until code inspection.

Recommendation:

- Track closely. If code is usable, REALM may become one of the strongest evaluation candidates for VLA/MLLM robustness.
- Do not integrate before a narrow install and action API spike.

### RoboTwin 2.0

Official source: [project page](https://robotwin-platform.github.io/).

RoboTwin 2.0 provides data generation and benchmarking for bimanual manipulation, with 50 dual-arm tasks, five robot embodiments, a large object dataset, and domain randomization across clutter, lighting, background, tabletop height, and language.

Fit:

- Task: high for bimanual manipulation.
- Vision/randomization: high.
- Control: low-medium for current code because the benchmark currently assumes one virtual joystick/action interface, not two coordinated arms.
- uv: unknown-medium, likely heavy.

Recommendation:

- Long-term target after extending `ControlInterfaceConfig` and logging to multi-arm action schemas.
- Useful now as a benchmark-validity and domain-randomization reference.

### RoboVerse / MetaSim

Official source: [project page](https://roboverseorg.github.io/).

RoboVerse proposes a simulation platform, synthetic dataset, and unified benchmarks. Its MetaSim component abstracts multiple simulation environments behind a universal interface.

Fit:

- Architectural relevance: high.
- Direct backend value: medium, because this project already has its own simulator boundary.
- uv: unknown.

Recommendation:

- Watch for reusable adapter ideas.
- Avoid adopting MetaSim as a dependency unless it clearly reduces integration cost across several future backends.

### Isaac Lab

Official source: [documentation](https://isaac-sim.github.io/IsaacLab/main/).

Isaac Lab is a modular robot learning framework built on NVIDIA Isaac Sim. It supports robot learning workflows, sensors, task creation, fast simulation, tiled rendering, domain randomization, and many robot assets.

Fit:

- Vision/physics: high.
- Task: medium-high, depending on selected tasks.
- Control: high in principle because Isaac Lab has controllers and task-space control docs.
- uv: low-medium. Isaac Sim and Isaac Lab are large GPU/runtime stacks and may be better managed with containers or a dedicated machine profile than a normal lightweight uv venv.

Recommendation:

- Treat as a long-term high-fidelity backend.
- Do not block near-term benchmark progress on Isaac unless a specific high-fidelity or GPU-parallel requirement appears.

### ARNOLD

Official source: [project page](https://arnold-benchmark.github.io/).

ARNOLD is an Isaac Sim benchmark for language-grounded tasks with continuous object states. It includes 8 tasks, 10k demonstrations, 20 scenes, 40 objects, multiple cameras, and continuous goal-state splits.

Fit:

- Task: high for continuous-state manipulation.
- Vision: high.
- Control: likely high but Isaac-specific.
- uv: low-medium due to Isaac.

Recommendation:

- Use as an Isaac-based benchmark reference.
- If Isaac Lab integration happens, ARNOLD can be a focused downstream benchmark candidate.

### BEHAVIOR-1K / OmniGibson

Official source: [BEHAVIOR documentation](https://behavior.stanford.edu/).

BEHAVIOR-1K and OmniGibson target 1,000 everyday activities in rich scenes with objects, object states, controllers, sensors, metrics, and teleoperation tooling. This is highly relevant to embodied AI, but it is also a large simulator ecosystem.

Fit:

- Task/scene: high.
- Vision: high.
- Control: medium. Needs local verification for low-level EEF joystick mapping.
- uv: low. Large assets and Isaac/Omniverse-style dependencies increase environment risk.

Recommendation:

- Keep as a long-term household benchmark.
- Avoid as the next adapter unless the project explicitly shifts toward full household activities beyond tabletop manipulation.

### AI2-THOR / ManipulaTHOR

Official sources: [AI2-THOR](https://ai2thor.allenai.org/) and [ManipulaTHOR](https://ai2thor.allenai.org/manipulathor/).

AI2-THOR provides Unity-based interactive indoor environments. ManipulaTHOR adds visual manipulation with a robotic arm, egocentric RGB-D and touch sensors, mobile manipulation, and 6-DOF manipulation.

Fit:

- Vision/interactive scene: high.
- Task: medium for this project; strong if navigation plus manipulation matters.
- Control: medium-low. The API is not obviously aligned with this repository's low-level delta EEF joystick layer.
- uv: medium-low due to Unity runtime/headless rendering considerations.

Recommendation:

- Not a near-term manipulation backend.
- Reconsider if benchmark scope expands to mobile manipulation and object-search workflows.

### Habitat Lab

Official source: [Habitat Lab docs](https://aihabitat.org/docs/habitat-lab/).

Habitat Lab is a modular high-level library for embodied AI across tasks, environments, and simulators. It is valuable for navigation and rearrangement research, but it is less directly aligned with low-level robot-arm teleoperation than RoboCasa, RLBench, or ManiSkill-family environments.

Recommendation:

- Not a near-term backend for the current joystick manipulation benchmark.

### VIMA-Bench

Official source: [project page](https://vimalabs.github.io/).

VIMA-Bench is a multimodal-prompt tabletop manipulation benchmark with 17 representative tasks, procedurally generated instances, and 600k+ expert trajectories. It is useful because it frames tasks as multimodal prompts, which overlaps with MLLM evaluation.

Fit:

- Prompt/task design: high.
- Control: medium-low. It is more naturally a prompt/action-token benchmark than a low-level virtual joystick environment.
- uv: unknown-medium.

Recommendation:

- Use VIMA for prompt and task-suite design inspiration.
- Do not prioritize it as a simulator backend unless its action interface can be made closed-loop and joystick-compatible without distorting the benchmark.

### Ravens / CLIPort

Official sources: [Ravens GitHub](https://github.com/google-research/ravens) and [CLIPort project page](https://cliport.github.io/).

Ravens provides 10 PyBullet tabletop rearrangement tasks with a Gym-like API, scripted oracles, reward functions, RGB-D observations, and primitive pick/place actions. The GitHub repository is archived as of 2025-10-02. CLIPort builds on this style for language-conditioned tabletop manipulation.

Fit:

- Task: medium for simple tabletop baselines.
- Control: low-medium because primitive pick/place actions are not the same as joystick delta control.
- uv: low-medium due to old Python/TensorFlow/PyBullet stack.

Recommendation:

- Do not adopt as a new backend unless a simple PyBullet baseline is needed.
- The archived status makes it less attractive than Meta-World for lightweight breadth.

## uv Feasibility Summary

| Candidate | uv feasibility | Notes |
| --- | --- | --- |
| Meta-World | High | Normal Python package path; Gymnasium API. |
| RoboCasa | Medium | Editable source installs and large assets; robosuite commit must be pinned. |
| SimplerEnv | Medium-low | Submodules, SAPIEN/ManiSkill2 lineage, CUDA requirements. |
| RLBench | Medium-low | Python can be isolated, but CoppeliaSim is an external binary/runtime. |
| Colosseum | Medium-low | Likely inherits RLBench-style external runtime cost. |
| LIBERO | Low-medium | Older Python/Torch/CUDA stack; may need a very old locked uv environment or fallback. |
| VLABench/GemBench/REALM | Unknown-medium | Need code-level install spikes. |
| RoboTwin 2.0 | Unknown-medium | Need dual-arm action schema and dependency inspection. |
| Isaac Lab / ARNOLD | Low-medium | Better treated as an Isaac machine/container profile than a simple uv-only backend. |
| BEHAVIOR-1K / OmniGibson | Low | Heavy simulator/assets; use dedicated environment strategy. |
| AI2-THOR / ManipulaTHOR | Medium-low | Python package plus Unity runtime/headless constraints. |
| Habitat Lab | Medium | Install may be manageable, but task fit is not strong enough. |
| Ravens / CLIPort | Low-medium | Archived/older stack; not worth dependency risk unless needed. |

## Benchmark-Safety Requirements for Any New Candidate

Before a new environment is treated as a valid benchmark backend, add the following gates:

1. **Adapter smoke test**: import backend lazily, create one environment, reset with seed, step one deterministic action, render one RGB frame, and close.
2. **Action contract test**: assert action dimensionality, gripper placement, clipping, and controller-layer mapping/sign/frame transforms before simulator `step`.
3. **Observation contract test**: assert image shape, camera metadata presence when available, and stable state keys for EEF/object/target.
4. **Success metric test**: verify that the adapter exposes a task success signal that cannot be confused with timeout or truncation.
5. **Randomization test**: verify camera/visual/task randomization through runtime API or explicit reset parameters, not static asset mutation.
6. **No-shortcut baseline**: run random, scripted, and where relevant no-language baselines before claiming MLLM ability.
7. **Statistical reporting**: report seed count, confidence interval or posterior interval, and exact perturbation distribution.
8. **Dependency isolation**: one `envs/uv/<backend>.in`, one lock file, no optional simulator imports in core package import path.

## Proposed Implementation Roadmap

### Phase 1: Lowest-risk backend breadth

- Done: add `metaworld` uv input/lock and one MT1 `reach-v3` adapter over the Gymnasium API.
- Done: gate import/reset/step/render and one benchmark episode with trajectory logging through `tests/test_metaworld_integration.py`.
- Next: add a task selector only after action/state/success fields are stable across a second MT1 task.
- Next: defer MT10/MT50 vector-env support until the single-env contract has real benchmark coverage.

### Phase 2: Semantically rich household manipulation

- Spike `robocasa` in a separate uv environment.
- Pin robosuite and RoboCasa commits instead of tracking master.
- Download assets outside tests and document the asset path.
- Implement one short-horizon RoboCasa task through a dedicated adapter.
- Reuse existing robosuite controller concepts only after confirming action-space equivalence.

### Phase 3: Real-to-sim/VLA evaluation

- Spike SimplerEnv on the GPU station.
- Verify the 7D action contract against this project's full joystick interface.
- Add a smoke test that can skip cleanly without GPU/SAPIEN assets.
- Use it for policy-behavior and perturbation studies rather than broad task coverage.

### Phase 4: CoppeliaSim family

- Install CoppeliaSim and PyRep outside uv, then create an RLBench uv env that points to the external runtime.
- Implement one RLBench task with one pinned action mode.
- Add Colosseum only after RLBench is reliable.

### Phase 5: High-fidelity long-term systems

- Evaluate Isaac Lab, ARNOLD, BEHAVIOR-1K/OmniGibson, REALM, and RoboTwin 2.0 as larger workstreams.
- Require a separate design note before implementation because each may force changes to action schemas, asset management, runtime launch, or GPU/container assumptions.

## Source Index

- Project design: [benchmark_design.md](benchmark_design.md), [architecture.md](architecture.md), [environment_management.md](environment_management.md).
- Meta-World: [project page](https://meta-world.github.io/), [GitHub](https://github.com/Farama-Foundation/Metaworld).
- RoboCasa: [project page](https://robocasa.ai/), [GitHub](https://github.com/robocasa/robocasa).
- SimplerEnv: [project page](https://simpler-env.github.io/), [GitHub](https://github.com/simpler-env/SimplerEnv).
- RLBench: [GitHub](https://github.com/stepjam/RLBench), [paper](https://arxiv.org/abs/1909.12271).
- The Colosseum: [project page](https://robot-colosseum.github.io/).
- LIBERO: [project page](https://libero-project.github.io/), [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO).
- VLABench: [project page](https://vlabench.github.io/).
- GemBench: [project page](https://www.di.ens.fr/willow/research/gembench/).
- REALM: [project page](https://martin-sedlacek.com/realm/), [paper](https://arxiv.org/abs/2512.19562).
- RoboTwin 2.0: [project page](https://robotwin-platform.github.io/).
- RoboVerse: [project page](https://roboverseorg.github.io/).
- Isaac Lab: [documentation](https://isaac-sim.github.io/IsaacLab/main/).
- ARNOLD: [project page](https://arnold-benchmark.github.io/).
- BEHAVIOR-1K / OmniGibson: [documentation](https://behavior.stanford.edu/).
- AI2-THOR / ManipulaTHOR: [AI2-THOR](https://ai2thor.allenai.org/), [ManipulaTHOR](https://ai2thor.allenai.org/manipulathor/).
- Habitat Lab: [documentation](https://aihabitat.org/docs/habitat-lab/).
- VIMA-Bench: [project page](https://vimalabs.github.io/).
- Ravens / CLIPort: [Ravens GitHub](https://github.com/google-research/ravens), [CLIPort project page](https://cliport.github.io/).
- Benchmark-validity audit: [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233).
