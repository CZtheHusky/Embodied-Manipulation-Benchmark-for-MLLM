from __future__ import annotations


from mllm_teleop_benchmark.agents.base import Agent, AgentContext
from mllm_teleop_benchmark.core.types import AgentDecision, AgentTrace


class ScriptedPolicy(Agent):
    """Simple reach-target policy for the dummy environment and fixed mapping."""

    def act(self, context: AgentContext) -> AgentDecision:
        state = context.observation.state
        ee = tuple(float(v) for v in state["ee_position"])
        target = tuple(float(v) for v in state["target_position"])
        channels = {name: 0.0 for name in context.action_space.channels}
        axis_channels = ("joystick_0", "joystick_1", "joystick_2")
        for idx, channel in enumerate(axis_channels):
            if channel not in channels:
                continue
            diff = target[idx] - ee[idx]
            if abs(diff) > 1e-9:
                channels[channel] = 1.0 if diff > 0 else -1.0
        for channel in ("joystick_3", "joystick_4", "joystick_5"):
            if channel in channels:
                channels[channel] = 0.0
        if "joystick_6" in channels:
            channels["joystick_6"] = 0.0
        return AgentDecision(action=channels, trace=AgentTrace(actor_type="scripted"))

    def metadata(self) -> dict[str, object]:
        return {"type": "scripted"}
