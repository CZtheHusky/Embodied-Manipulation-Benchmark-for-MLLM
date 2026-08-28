from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any


@dataclass(frozen=True)
class BenchmarkSection:
    name: str
    num_episodes: int
    seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSection":
        return cls(
            name=str(data["name"]),
            num_episodes=int(data["num_episodes"]),
            seed=int(data["seed"]),
        )


@dataclass(frozen=True)
class SimulatorConfig:
    type: str
    control_dt: float = 0.1
    workspace_low: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_high: tuple[float, float, float] = (1.0, 1.0, 1.0)
    env_id: str | None = None
    render_mode: str | None = None
    frame_skip: int | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    wrappers: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulatorConfig":
        bounds = data.get("workspace_bounds", {})
        low = tuple(float(v) for v in bounds.get("low", (-1.0, -1.0, 0.0)))
        high = tuple(float(v) for v in bounds.get("high", (1.0, 1.0, 1.0)))
        return cls(
            type=str(data["type"]),
            control_dt=float(data.get("control_dt", 0.1)),
            workspace_low=low,
            workspace_high=high,
            env_id=str(data["env_id"]) if "env_id" in data else None,
            render_mode=str(data["render_mode"]) if data.get("render_mode") is not None else None,
            frame_skip=int(data["frame_skip"]) if "frame_skip" in data else None,
            env_kwargs=dict(data.get("env_kwargs", {})),
            wrappers=tuple(str(v) for v in data.get("wrappers", ())),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class TaskConfig:
    type: str
    max_steps: int
    success_threshold: float
    target_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_ee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_hold_steps: int = 1
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskConfig":
        target = tuple(float(v) for v in data.get("target_position", (0.0, 0.0, 0.0)))
        initial = tuple(float(v) for v in data.get("initial_ee_position", (0.0, 0.0, 0.0)))
        return cls(
            type=str(data["type"]),
            max_steps=int(data["max_steps"]),
            success_threshold=float(data["success_threshold"]),
            target_position=target,
            initial_ee_position=initial,
            goal_hold_steps=int(data.get("goal_hold_steps", 1)),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class CameraSpec:
    name: str
    type: str
    width: int
    height: int
    pose: tuple[float, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSpec":
        return cls(
            name=str(data["name"]),
            type=str(data.get("type", "dummy_rgb")),
            width=int(data.get("width", 512)),
            height=int(data.get("height", 512)),
            pose=tuple(float(v) for v in data.get("pose", ())),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class CameraConfig:
    manager: str
    visible_metadata: bool
    cameras: tuple[CameraSpec, ...]
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraConfig":
        return cls(
            manager=str(data.get("manager", "fixed")),
            visible_metadata=bool(data.get("visible_metadata", True)),
            cameras=tuple(CameraSpec.from_dict(item) for item in data["cameras"]),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class ChannelConfig:
    kind: str
    values: tuple[float, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelConfig":
        return cls(
            kind=str(data["kind"]),
            values=tuple(float(v) for v in data.get("values", ())),
            minimum=float(data["min"]) if "min" in data else None,
            maximum=float(data["max"]) if "max" in data else None,
        )


@dataclass(frozen=True)
class ControlInterfaceConfig:
    mode: str
    active_dimensions: tuple[str, ...]
    gripper: str = "none"
    channel_kind: str = "discrete"
    channel_values: tuple[float, ...] = (-1.0, 0.0, 1.0)
    channel_minimum: float | None = None
    channel_maximum: float | None = None
    mapping_strategy: str = "fixed"
    visible_metadata: bool = True
    mapping_visible_to_agent: bool = True
    frame_transform: str = "identity"
    frame_visible_to_agent: bool = True
    control_frame_description: str | None = None
    dimension_semantics_description: str | None = None
    gripper_semantics_description: str | None = None
    gripper_open_command: float | None = None
    gripper_closed_command: float | None = None
    gripper_hold_command: float | None = None
    gripper_neutral_command: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "control_frame_description",
            "dimension_semantics_description",
            "gripper_semantics_description",
        ):
            raw_description = getattr(self, field_name)
            if raw_description is None:
                continue
            description = str(raw_description).strip()
            object.__setattr__(self, field_name, description)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlInterfaceConfig":
        active_dimensions = tuple(str(v) for v in data["active_dimensions"])
        gripper = str(data.get("gripper", "none"))
        channel_kind = str(data.get("channel_kind", "discrete"))
        return cls(
            mode=str(data.get("mode", "eef_delta_pose")),
            active_dimensions=active_dimensions,
            gripper=gripper,
            channel_kind=channel_kind,
            channel_values=tuple(float(v) for v in data.get("channel_values", (-1.0, 0.0, 1.0))),
            channel_minimum=float(data["channel_min"]) if "channel_min" in data else None,
            channel_maximum=float(data["channel_max"]) if "channel_max" in data else None,
            mapping_strategy=str(data.get("mapping_strategy", "fixed")),
            visible_metadata=bool(data.get("visible_metadata", True)),
            mapping_visible_to_agent=bool(data.get("mapping_visible_to_agent", data.get("visible_metadata", True))),
            frame_transform=str(data.get("frame_transform", "identity")),
            frame_visible_to_agent=bool(data.get("frame_visible_to_agent", data.get("visible_metadata", True))),
            control_frame_description=(
                str(data["control_frame_description"])
                if "control_frame_description" in data
                else None
            ),
            dimension_semantics_description=(
                str(data["dimension_semantics_description"])
                if "dimension_semantics_description" in data
                else None
            ),
            gripper_semantics_description=(
                str(data["gripper_semantics_description"])
                if "gripper_semantics_description" in data
                else None
            ),
            gripper_open_command=(
                float(data["gripper_open_command"])
                if "gripper_open_command" in data
                else None
            ),
            gripper_closed_command=(
                float(data["gripper_closed_command"])
                if "gripper_closed_command" in data
                else None
            ),
            gripper_hold_command=(
                float(data["gripper_hold_command"])
                if "gripper_hold_command" in data
                else None
            ),
            gripper_neutral_command=(
                float(data["gripper_neutral_command"])
                if "gripper_neutral_command" in data
                else None
            ),
            params=dict(data.get("params", {})),
        )

    def channel_names(self) -> tuple[str, ...]:
        names = [f"joystick_{idx}" for idx, _ in enumerate(self.active_dimensions)]
        if self.gripper != "none":
            names.append(f"joystick_{len(names)}")
        return tuple(names)

    def control_dimensions(self) -> tuple[str, ...]:
        dims = list(self.active_dimensions)
        if self.gripper != "none":
            dims.append("gripper")
        return tuple(dims)

    def to_action_space_config(self) -> "ActionSpaceConfig":
        channels: dict[str, ChannelConfig] = {}
        motion_channel = {
            "discrete": ChannelConfig(kind="discrete", values=self.channel_values),
            "continuous": ChannelConfig(
                kind="continuous",
                minimum=self.channel_minimum,
                maximum=self.channel_maximum,
            ),
        }[self.channel_kind]
        for name, dim in zip(self.channel_names(), self.control_dimensions(), strict=True):
            if dim == "gripper" and self.gripper in {"direct", "effort"}:
                channels[name] = ChannelConfig(kind="continuous", minimum=0.0, maximum=1.0)
            elif dim == "gripper" and self.gripper == "binary":
                channels[name] = ChannelConfig(kind="discrete", values=(0.0, 1.0))
            elif dim == "gripper" and self.gripper == "ternary":
                middle = (
                    self.gripper_hold_command
                    if self.gripper_hold_command is not None
                    else self.gripper_neutral_command
                )
                channels[name] = ChannelConfig(
                    kind="discrete",
                    values=(
                        float(self.gripper_open_command),
                        float(middle),
                        float(self.gripper_closed_command),
                    ),
                )
            else:
                channels[name] = motion_channel
        return ActionSpaceConfig(type="joystick", channels=channels)

    def to_mapping_config(self, seed: int | None = None) -> "MappingConfig":
        permuted, sign_flipped = {
            "fixed": (False, False),
            "permuted": (True, False),
            "sign_flipped": (False, True),
            "permuted_sign_flipped": (True, True),
        }[self.mapping_strategy]
        motion_channels = list(self.channel_names()[: len(self.active_dimensions)])
        motion_dims = list(self.active_dimensions)
        rng = random.Random(seed if seed is not None else 0)
        if permuted:
            rng.shuffle(motion_dims)
            if (
                self.mapping_strategy == "permuted_sign_flipped"
                and len(motion_dims) > 1
                and motion_dims == list(self.active_dimensions)
            ):
                motion_dims[0], motion_dims[1] = motion_dims[1], motion_dims[0]
        channels = dict(zip(motion_channels, motion_dims, strict=True))
        signs = {channel: 1.0 for channel in motion_channels}
        if sign_flipped:
            signs = {channel: float(rng.choice((-1.0, 1.0))) for channel in motion_channels}
            if (
                self.mapping_strategy == "permuted_sign_flipped"
                and motion_channels
                and all(sign > 0.0 for sign in signs.values())
            ):
                signs[motion_channels[0]] = -1.0
        if self.gripper != "none":
            gripper_channel = self.channel_names()[-1]
            channels[gripper_channel] = "gripper"
            signs[gripper_channel] = 1.0
        return MappingConfig(
            type="fixed",
            visible_to_agent=self.mapping_visible_to_agent,
            channels=channels,
            signs=signs,
        )

    def metadata(self) -> dict[str, Any]:
        if not self.visible_metadata:
            return {
                "mode": self.mode,
                "joystick_channels": list(self.channel_names()),
                "visible_metadata": False,
            }
        visible = {
            "mode": self.mode,
            "active_dimensions": list(self.active_dimensions),
            "gripper": self.gripper,
            "joystick_channels": list(self.channel_names()),
            "visible_metadata": self.visible_metadata,
            "scale_mode": self.params.get("scale_mode", "configured_by_controller"),
            "mapping_complexity_present": self.mapping_strategy != "fixed",
            "frame_complexity_present": self.frame_transform != "identity",
        }
        if self.control_frame_description:
            visible["control_frame_description"] = self.control_frame_description
        if self.dimension_semantics_description:
            visible["dimension_semantics_description"] = (
                self.dimension_semantics_description
            )
        if self.gripper_semantics_description:
            visible["gripper_semantics_description"] = (
                self.gripper_semantics_description
            )
        rotation_representation = self.params.get("rotation_representation")
        if rotation_representation is not None:
            visible["rotation_representation"] = str(rotation_representation)
        gripper_commands = {
            name: value
            for name, value in (
                ("open", self.gripper_open_command),
                ("closed", self.gripper_closed_command),
                ("hold", self.gripper_hold_command),
                ("neutral", self.gripper_neutral_command),
            )
            if value is not None
        }
        if gripper_commands:
            visible["gripper_commands"] = gripper_commands
        return visible


@dataclass(frozen=True)
class ActionSpaceConfig:
    type: str
    channels: dict[str, ChannelConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpaceConfig":
        channels = {
            name: ChannelConfig.from_dict(value)
            for name, value in data["channels"].items()
        }
        return cls(type=str(data.get("type", "joystick")), channels=channels)


@dataclass(frozen=True)
class MappingConfig:
    type: str
    visible_to_agent: bool
    channels: dict[str, str]
    signs: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MappingConfig":
        return cls(
            type=str(data.get("type", "fixed")),
            visible_to_agent=bool(data.get("visible_to_agent", True)),
            channels=dict(data["channels"]),
            signs={str(k): float(v) for k, v in data.get("signs", {}).items()},
        )


@dataclass(frozen=True)
class ControllerConfig:
    type: str
    frame: str
    scales: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerConfig":
        return cls(
            type=str(data["type"]),
            frame=str(data.get("frame", "base")),
            scales={k: float(v) for k, v in data.get("scales", {}).items()},
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class FrameTransformConfig:
    type: str
    visible_to_agent: bool
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameTransformConfig":
        return cls(
            type=str(data.get("type", "identity")),
            visible_to_agent=bool(data.get("visible_to_agent", True)),
            params=dict(data.get("params", {})),
        )


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "openai_compatible"
    api: str = "chat_completions"
    name: str | None = None
    name_env: str = "MLLM_TELEOP_MODEL"
    base_url: str | None = None
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    require_api_key: bool = True
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    retry_backoff_max_seconds: float = 30.0
    retry_until_success: bool = False
    reasoning_effort: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 128
    response_format: str = "json_object"
    prompt_cache_key: str | None = None
    stream: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        values = dict(data or {})
        # EpisodeLogger snapshots raw config values, so accepting this field would persist the secret.
        if "api_key" in values:
            raise ValueError("agent.model.api_key is forbidden; use agent.model.api_key_env")
        timeout_seconds = float(values.get("timeout_seconds", 60.0))
        max_retries = int(values.get("max_retries", 2))
        retry_backoff_seconds = float(values.get("retry_backoff_seconds", 1.0))
        retry_backoff_max_seconds = float(values.get("retry_backoff_max_seconds", 30.0))
        max_output_tokens = int(values.get("max_output_tokens", 128))
        response_format = str(values.get("response_format", "json_object"))
        api = str(values.get("api", "chat_completions"))
        reasoning_effort_value = values.get("reasoning_effort")
        reasoning_effort = (
            str(reasoning_effort_value) if reasoning_effort_value is not None else None
        )
        name = values.get("name")
        base_url = values.get("base_url")
        prompt_cache_key = values.get("prompt_cache_key")
        return cls(
            provider=str(values.get("provider", "openai_compatible")),
            api=api,
            name=str(name) if name is not None else None,
            name_env=str(values.get("name_env", "MLLM_TELEOP_MODEL")),
            base_url=str(base_url) if base_url is not None else None,
            base_url_env=str(values.get("base_url_env", "OPENAI_BASE_URL")),
            api_key_env=str(values.get("api_key_env", "OPENAI_API_KEY")),
            require_api_key=bool(values.get("require_api_key", True)),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_backoff_max_seconds=retry_backoff_max_seconds,
            retry_until_success=bool(values.get("retry_until_success", False)),
            reasoning_effort=reasoning_effort,
            temperature=float(values.get("temperature", 0.0)),
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            prompt_cache_key=str(prompt_cache_key) if prompt_cache_key is not None else None,
            stream=bool(values.get("stream", False)),
        )


@dataclass(frozen=True)
class AgentObservationConfig:
    cameras: tuple[str, ...] = ()
    image_max_side: int = 512

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgentObservationConfig":
        values = dict(data or {})
        return cls(
            cameras=tuple(str(name) for name in values.get("cameras", ())),
            image_max_side=int(values.get("image_max_side", 512)),
        )


HistoryWindow = int | str


@dataclass(frozen=True)
class AgentHistoryConfig:
    action_window: HistoryWindow = 2
    image_window: HistoryWindow = 2
    max_images_per_request: int = 128
    max_request_bytes: int = 67_108_864

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgentHistoryConfig":
        values = dict(data or {})
        image_window = values.get("image_window", 2)
        return cls(
            action_window=values.get("action_window", image_window),
            image_window=image_window,
            max_images_per_request=int(values.get("max_images_per_request", 128)),
            max_request_bytes=int(values.get("max_request_bytes", 67_108_864)),
        )



@dataclass(frozen=True)
class AgentConfig:
    type: str
    invalid_action_recovery: str = "noop"
    history_window: int = 5
    model: ModelConfig | None = None
    observation: AgentObservationConfig = field(default_factory=AgentObservationConfig)
    history: AgentHistoryConfig = field(default_factory=AgentHistoryConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        agent_type = str(data["type"])
        invalid_action_recovery = str(data.get("invalid_action_recovery", "noop"))
        history_window = int(data.get("history_window", 5))
        # The removed field would otherwise be ignored and silently select the default API backend.
        if "action_output_mode" in data:
            raise ValueError(
                "agent.action_output_mode has been removed; select the backend with "
                "agent.model.api. Both chat_completions and responses accept optional "
                "Reflection."
            )
        model = ModelConfig.from_dict(data.get("model")) if agent_type == "mllm" else None
        history = AgentHistoryConfig.from_dict(data.get("history"))
        if (
            model is not None
            and model.api == "chat_completions"
            and history.action_window != history.image_window
        ):
            # CLI overrides can change only one window; without rejection, run metadata
            # records that unmatched value while the request builder emits paired turns.
            raise ValueError(
                "Chat Completions requires agent.history.action_window to equal "
                "agent.history.image_window"
            )
        return cls(
            type=agent_type,
            invalid_action_recovery=invalid_action_recovery,
            history_window=history_window,
            model=model,
            observation=AgentObservationConfig.from_dict(data.get("observation")),
            history=history,
        )


@dataclass(frozen=True)
class VideoConfig:
    fps: int = 2
    step_hold_seconds: float = 1.0
    terminal_hold_seconds: float = 2.0
    include_evaluator_debug: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VideoConfig":
        values = dict(data or {})
        fps = int(values.get("fps", 2))
        step_hold_seconds = float(values.get("step_hold_seconds", 1.0))
        terminal_hold_seconds = float(values.get("terminal_hold_seconds", 2.0))
        return cls(
            fps=fps,
            step_hold_seconds=step_hold_seconds,
            terminal_hold_seconds=terminal_hold_seconds,
            include_evaluator_debug=bool(values.get("include_evaluator_debug", False)),
        )


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: Path
    save_images: bool = True
    save_video: bool = False
    save_trajectory: bool = True
    save_config_snapshot: bool = True
    video: VideoConfig = field(default_factory=VideoConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggingConfig":
        save_video = bool(data.get("save_video", False))
        video = VideoConfig.from_dict(data.get("video"))
        # Without an MP4 there is no debug artifact to label, so this combination is invalid.
        if video.include_evaluator_debug and not save_video:
            raise ValueError(
                "logging.video.include_evaluator_debug=true requires logging.save_video=true"
            )
        return cls(
            output_dir=Path(str(data["output_dir"])),
            save_images=bool(data.get("save_images", True)),
            save_video=save_video,
            save_trajectory=bool(data.get("save_trajectory", True)),
            save_config_snapshot=bool(data.get("save_config_snapshot", True)),
            video=video,
        )


@dataclass(frozen=True)
class UIConfig:
    backend: str = "pygame"
    runtime: str = "sim_adapter_worker"
    window_width: int = 640
    window_height: int = 480
    info_panel_height: int = 170
    fps: int = 20
    max_steps: int | None = None
    step_when_idle: bool = False
    simple_blit: bool = True
    debug_frames: bool = False
    debug_control_history: bool = False
    frame_camera: str | None = None
    mujoco_gl: str | None = None
    display: str | None = None
    worker_timeout_seconds: float = 60.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "UIConfig":
        values = dict(data or {})
        max_steps = values.get("max_steps")
        frame_camera = values.get("frame_camera")
        mujoco_gl = values.get("mujoco_gl")
        display = values.get("display")
        return cls(
            backend=str(values.get("backend", "pygame")),
            runtime=str(values.get("runtime", "sim_adapter_worker")),
            window_width=int(values.get("window_width", 640)),
            window_height=int(values.get("window_height", 480)),
            info_panel_height=int(values.get("info_panel_height", 170)),
            fps=int(values.get("fps", 20)),
            max_steps=int(max_steps) if max_steps is not None else None,
            step_when_idle=bool(values.get("step_when_idle", False)),
            simple_blit=bool(values.get("simple_blit", True)),
            debug_frames=bool(values.get("debug_frames", False)),
            debug_control_history=bool(values.get("debug_control_history", False)),
            frame_camera=str(frame_camera) if frame_camera is not None else None,
            mujoco_gl=str(mujoco_gl) if mujoco_gl is not None else None,
            display=str(display) if display is not None else None,
            worker_timeout_seconds=float(values.get("worker_timeout_seconds", 60.0)),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark: BenchmarkSection
    simulator: SimulatorConfig
    task: TaskConfig
    camera: CameraConfig
    action_space: ActionSpaceConfig
    mapping: MappingConfig
    controller: ControllerConfig
    frame_transform: FrameTransformConfig
    agent: AgentConfig
    logging: LoggingConfig
    ui: UIConfig
    control_interface: ControlInterfaceConfig
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        source_path: str | None = None,
    ) -> "BenchmarkConfig":
        control_interface_data = data["control_interface"]
        control_interface = ControlInterfaceConfig.from_dict(control_interface_data)
        action_space = control_interface.to_action_space_config()
        mapping = control_interface.to_mapping_config()
        agent = AgentConfig.from_dict(data["agent"])
        logging = LoggingConfig.from_dict(data["logging"])
        # ReviewVideoRenderer consumes model artifacts that human agents never produce.
        if logging.save_video and agent.type != "mllm":
            raise ValueError("logging.save_video currently supports only agent.type=mllm")
        benchmark = BenchmarkSection.from_dict(data["benchmark"])
        task = TaskConfig.from_dict(data["task"])
        camera = CameraConfig.from_dict(data["camera"])
        return cls(
            benchmark=benchmark,
            simulator=SimulatorConfig.from_dict(data["simulator"]),
            task=task,
            camera=camera,
            action_space=action_space,
            mapping=mapping,
            controller=ControllerConfig.from_dict(data["controller"]),
            frame_transform=FrameTransformConfig.from_dict(
                {
                    "type": control_interface.frame_transform,
                    "visible_to_agent": control_interface.frame_visible_to_agent,
                    "params": control_interface.params.get("frame_transform", {}),
                }
            ),
            agent=agent,
            logging=logging,
            ui=UIConfig.from_dict(data.get("ui", {})),
            control_interface=control_interface,
            source_path=source_path,
            raw=data,
        )
