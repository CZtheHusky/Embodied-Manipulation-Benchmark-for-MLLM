from __future__ import annotations

from typing import Any

import numpy as np

from mllm_teleop_benchmark.config.schema import CameraSpec, SimulatorConfig, TaskConfig
from mllm_teleop_benchmark.core.types import (
    LanguageTableAction,
    Observation,
    SimulatorAction,
    StepResult,
)
from mllm_teleop_benchmark.sim.adapter import SimAdapter


class LanguageTableSimAdapter(SimAdapter):
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.env: Any | None = None
        self.step_id = 0
        self.last_native_observation: dict[str, Any] | None = None
        self.last_reward: float | None = None
        self.last_success = False

    def reset(self, task_config: TaskConfig, seed: int) -> Observation:
        self._ensure_env(seed)
        self.env.seed(seed)
        self.step_id = 0
        self.last_reward = None
        self.last_success = False
        self.last_native_observation = self.env.reset()
        return self._observation()

    def step(self, action: SimulatorAction) -> StepResult:
        # This branch selects the only simulator action representation supported by this adapter.
        if not isinstance(action, LanguageTableAction):
            raise TypeError(
                "LanguageTableSimAdapter requires LanguageTableAction, got "
                f"{type(action).__name__}"
            )
        native_action = np.asarray(action.delta_xy, dtype=np.float32)
        observation, reward, done, _ = self.env.step(native_action)
        self.step_id += 1
        self.last_native_observation = observation
        self.last_reward = float(reward)
        self.last_success = bool(done or reward > 0.0)
        return StepResult(
            observation=self._observation(),
            terminated=bool(done),
            truncated=False,
            info=self.get_state(),
        )

    def render(self, camera: CameraSpec) -> Any:
        # reset() or step() always supplies the native RGB used by the official environment.
        assert self.last_native_observation is not None
        return np.asarray(self.last_native_observation["rgb"], dtype=np.uint8).copy()

    def get_state(self) -> dict[str, Any]:
        # EpisodeManager calls get_state only after reset has established the environment state.
        assert self.env is not None and self.last_native_observation is not None
        native = self.last_native_observation
        state: dict[str, Any] = {
            "step_id": self.step_id,
            "native_step_id": self.step_id,
            "reward": self.last_reward,
            "success": self.last_success,
            "instruction": self.env.decode_instruction(native["instruction"]),
            "effector_position": np.asarray(
                native["effector_translation"], dtype=np.float32
            ).tolist(),
            "effector_target_position": np.asarray(
                native["effector_target_translation"], dtype=np.float32
            ).tolist(),
        }
        distance = self._object_target_distance()
        if distance is not None:
            state["object_target_distance"] = distance
            state["distance_to_target"] = distance
        return state

    def close(self) -> None:
        if self.env is None:
            return
        self.env.pybullet_client.disconnect()
        self.env = None
        self.last_native_observation = None

    def _ensure_env(self, seed: int) -> None:
        if self.env is not None:
            return
        from language_table.environments import blocks
        from language_table.environments import language_table
        from language_table.environments.rewards import block2absolutelocation
        from language_table.environments.rewards import block2block
        from language_table.environments.rewards import block2block_relative_location
        from language_table.environments.rewards import block2relativelocation
        from language_table.environments.rewards import separate_blocks

        reward_factories = {
            "blocktoblock": block2block.BlockToBlockReward,
            "blocktoabsolutelocation": (
                block2absolutelocation.BlockToAbsoluteLocationReward
            ),
            "blocktoblockrelativelocation": (
                block2block_relative_location.BlockToBlockRelativeLocationReward
            ),
            "blocktorelativelocation": (
                block2relativelocation.BlockToRelativeLocationReward
            ),
            "separate": separate_blocks.SeparateBlocksReward,
        }
        block_mode = blocks.LanguageTableBlockVariants[
            str(self.config.params.get("block_mode", "BLOCK_4"))
        ]
        reward_factory = reward_factories[
            str(self.config.params.get("reward", "blocktoblock"))
        ]
        self.env = language_table.LanguageTable(
            block_mode=block_mode,
            training=bool(self.config.params.get("training", True)),
            reward_factory=reward_factory,
            control_frequency=float(self.config.params.get("control_frequency", 10.0)),
            shared_memory=False,
            seed=seed,
            delay_reward_steps=int(self.config.params.get("delay_reward_steps", 0)),
            render_text_in_image=False,
            debug_visuals=False,
            add_invisible_walls=bool(
                self.config.params.get("add_invisible_walls", False)
            ),
        )

    def _object_target_distance(self) -> float | None:
        start_block = self.env._start_block
        target_block = self.env._oracle_target_block
        target_translation = self.env._oracle_target_translation
        block_ids = self.env._block_to_pybullet_id
        if start_block not in block_ids:
            return None
        start_position = np.asarray(
            self.env.pybullet_client.getBasePositionAndOrientation(
                block_ids[start_block]
            )[0][:2],
            dtype=np.float64,
        )
        if target_block is not None:
            target_position = np.asarray(
                self.env.pybullet_client.getBasePositionAndOrientation(
                    block_ids[target_block]
                )[0][:2],
                dtype=np.float64,
            )
        elif target_translation is not None:
            target_position = np.asarray(target_translation[:2], dtype=np.float64)
        else:
            return None
        return float(np.linalg.norm(start_position - target_position))

    def _observation(self) -> Observation:
        return Observation(
            step_id=self.step_id,
            images={},
            state=self.get_state(),
            metadata={
                "simulator": "language_table",
                "native_rgb_shape": [180, 320, 3],
            },
        )
