from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from mllm_teleop_benchmark.config.schema import BenchmarkConfig
from mllm_teleop_benchmark.core.types import EpisodeResult, Observation, StepTransition
from mllm_teleop_benchmark.utils.images import encode_rgb_png, safe_artifact_name
from mllm_teleop_benchmark.utils.serialization import to_jsonable


RENDERER_VERSION = "mllm_review_v1"
CANVAS_SIZE = (1280, 720)
IMAGE_REGION_WIDTH = 850
_ORACLE_DISTANCE_KEYS = ("distance_to_target", "goal_distance", "distance")
_ORACLE_TARGET_KEYS = ("target_position", "goal_position")
_ORACLE_EEF_KEYS = (
    "eef_position",
    "ee_position",
    "fingertip_position",
    "end_effector_position",
)


class ReviewVideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReviewVideoResult:
    status: str
    model_view: str | None
    evaluator_debug: str | None
    manifest: str
    error: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


class ReviewVideoRenderer:
    def __init__(
        self,
        config: BenchmarkConfig,
        episode_dir: Path,
        *,
        ffmpeg_binary: str = "ffmpeg",
    ) -> None:
        self.config = config
        self.episode_dir = episode_dir
        self.review_dir = episode_dir / "review"
        self.ffmpeg_binary = ffmpeg_binary

    def render(
        self,
        transitions: tuple[StepTransition, ...],
        terminal_observation: Observation,
        episode_metadata: dict[str, Any],
        episode_result: EpisodeResult,
    ) -> ReviewVideoResult:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.review_dir / "manifest.json"
        model_path = self.review_dir / "model_view.mp4"
        debug_path = self.review_dir / "evaluator_debug.mp4"
        manifest: dict[str, Any] = {
            "renderer_version": RENDERER_VERSION,
            "canvas": {"width": CANVAS_SIZE[0], "height": CANVAS_SIZE[1]},
            "fps": self.config.logging.video.fps,
            "step_hold_seconds": self.config.logging.video.step_hold_seconds,
            "terminal_hold_seconds": self.config.logging.video.terminal_hold_seconds,
            "ffmpeg": {"binary": self.ffmpeg_binary, "version": None},
            "steps": [],
            "terminal": None,
            "outputs": {},
            "status": "pending",
        }
        errors: list[dict[str, Any]] = []

        try:
            camera_order = self._camera_order(transitions, terminal_observation)
            step_records = [
                self._load_step_record(transition, camera_order, episode_metadata)
                for transition in transitions
            ]
            terminal_record = self._save_terminal_images(terminal_observation, camera_order)
            manifest["steps"] = step_records
            manifest["terminal"] = terminal_record
            manifest["ffmpeg"]["version"] = self._ffmpeg_version()
        except Exception as exc:
            errors.append(_error_record("prepare", exc))
            step_records = []
            terminal_record = None

        if not errors and terminal_record is not None:
            model_error = self._render_output(
                model_path,
                self._model_frames(step_records, terminal_record, episode_metadata),
            )
            manifest["outputs"]["model_view"] = _output_record(model_path, model_error)
            if model_error is not None:
                errors.append(model_error)

            if self.config.logging.video.include_evaluator_debug:
                debug_error = self._render_output(
                    debug_path,
                    self._debug_frames(
                        step_records,
                        terminal_record,
                        transitions,
                        terminal_observation,
                        episode_metadata,
                        episode_result,
                    ),
                )
                manifest["outputs"]["evaluator_debug"] = _output_record(debug_path, debug_error)
                if debug_error is not None:
                    errors.append(debug_error)
        elif "model_view" not in manifest["outputs"]:
            manifest["outputs"]["model_view"] = {"status": "not_attempted", "path": None}

        status = "failed" if errors else "completed"
        manifest["status"] = status
        if errors:
            manifest["errors"] = errors
        _write_json(manifest_path, manifest)

        error = None
        if errors:
            error = {"errors": errors}
            _write_json(self.review_dir / "video_error.json", error)

        return ReviewVideoResult(
            status=status,
            model_view=_relative_to_episode(model_path, self.episode_dir) if model_path.exists() else None,
            evaluator_debug=(
                _relative_to_episode(debug_path, self.episode_dir) if debug_path.exists() else None
            ),
            manifest=_relative_to_episode(manifest_path, self.episode_dir),
            error=error,
        )

    def _camera_order(
        self,
        transitions: tuple[StepTransition, ...],
        terminal_observation: Observation,
    ) -> tuple[str, ...]:
        configured = self.config.agent.observation.cameras
        if configured:
            cameras = tuple(configured)
        elif transitions:
            cameras = tuple(sorted(str(name) for name in transitions[0].observation.images))
        else:
            cameras = tuple(sorted(str(name) for name in terminal_observation.images))
        if not cameras:
            raise ReviewVideoError("Review video requires at least one public RGB camera")
        return cameras

    def _load_step_record(
        self,
        transition: StepTransition,
        camera_order: tuple[str, ...],
        episode_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        trace = transition.agent_trace
        if trace is None or not trace.artifact_dir:
            raise ReviewVideoError(f"Step {transition.step_id} has no model request artifact directory")
        request_path = self._artifact_path(f"{trace.artifact_dir}/request.json")
        try:
            request_manifest = json.loads(request_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReviewVideoError(f"Missing model request manifest: {request_path}") from exc
        except json.JSONDecodeError as exc:
            raise ReviewVideoError(f"Invalid model request manifest: {request_path}") from exc

        images = request_manifest.get("images")
        if not isinstance(images, list):
            raise ReviewVideoError(f"Model request manifest has no image list: {request_path}")
        sources = []
        for camera in camera_order:
            matches = [
                image
                for image in images
                if isinstance(image, dict)
                and image.get("step_id") == transition.step_id
                and image.get("camera") == camera
            ]
            if len(matches) != 1:
                raise ReviewVideoError(
                    f"Step {transition.step_id} camera {camera!r} has {len(matches)} current image records"
                )
            sources.append(self._validated_source(matches[0]))

        return {
            "step_id": transition.step_id,
            "images": sources,
            "model_record": model_review_record(transition, episode_metadata),
        }

    def _validated_source(self, image: dict[str, Any]) -> dict[str, Any]:
        artifact = image.get("artifact")
        expected_hash = image.get("sha256")
        if not isinstance(artifact, str) or not isinstance(expected_hash, str):
            raise ReviewVideoError("Model image manifest is missing artifact or sha256")
        path = self._artifact_path(artifact)
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise ReviewVideoError(f"Missing model-visible PNG: {path}") from exc
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise ReviewVideoError(
                f"Model-visible PNG hash mismatch for {artifact}: expected {expected_hash}, got {actual_hash}"
            )
        try:
            with Image.open(path) as frame:
                frame.verify()
        except Exception as exc:
            raise ReviewVideoError(f"Invalid model-visible PNG: {artifact}") from exc
        return {
            "camera": str(image.get("camera")),
            "artifact": artifact,
            "sha256": expected_hash,
            "width": int(image.get("width", 0)),
            "height": int(image.get("height", 0)),
        }

    def _save_terminal_images(
        self,
        observation: Observation,
        camera_order: tuple[str, ...],
    ) -> dict[str, Any]:
        sources = []
        terminal_dir = self.review_dir / "terminal"
        terminal_dir.mkdir(parents=True, exist_ok=True)
        for camera in camera_order:
            if camera not in observation.images:
                raise ReviewVideoError(f"Terminal observation is missing camera {camera!r}")
            try:
                encoded = encode_rgb_png(
                    observation.images[camera],
                    self.config.agent.observation.image_max_side,
                )
            except Exception as exc:
                raise ReviewVideoError(
                    f"Failed to encode terminal camera {camera!r}: {type(exc).__name__}"
                ) from exc
            relative = f"review/terminal/{safe_artifact_name(camera)}.png"
            target = self._artifact_path(relative)
            target.write_bytes(encoded.content)
            sources.append(
                {
                    "camera": camera,
                    "artifact": relative,
                    "sha256": encoded.sha256,
                    "width": encoded.width,
                    "height": encoded.height,
                }
            )
        return {"step_id": observation.step_id, "images": sources}

    def _model_frames(
        self,
        step_records: list[dict[str, Any]],
        terminal_record: dict[str, Any],
        episode_metadata: dict[str, Any],
    ) -> Iterable[tuple[Image.Image, float]]:
        for record in step_records:
            yield self._render_canvas(record["images"], record["model_record"], evaluator_only=False), (
                self.config.logging.video.step_hold_seconds
            )
        terminal_panel = {
            "task": str(episode_metadata.get("task_instruction", "")),
            "step": terminal_record["step_id"],
            "status": "terminal post-action observation",
        }
        yield self._render_canvas(
            terminal_record["images"],
            terminal_panel,
            evaluator_only=False,
        ), self.config.logging.video.terminal_hold_seconds

    def _debug_frames(
        self,
        step_records: list[dict[str, Any]],
        terminal_record: dict[str, Any],
        transitions: tuple[StepTransition, ...],
        terminal_observation: Observation,
        episode_metadata: dict[str, Any],
        episode_result: EpisodeResult,
    ) -> Iterable[tuple[Image.Image, float]]:
        for record, transition in zip(step_records, transitions, strict=True):
            panel = dict(record["model_record"])
            panel["evaluator"] = evaluator_debug_record(transition)
            yield self._render_canvas(record["images"], panel, evaluator_only=True), (
                self.config.logging.video.step_hold_seconds
            )
        terminal_panel = {
            "task": str(episode_metadata.get("task_instruction", "")),
            "step": terminal_record["step_id"],
            "status": "terminal post-action observation",
            "evaluator": evaluator_terminal_record(terminal_observation, episode_result),
        }
        yield self._render_canvas(
            terminal_record["images"],
            terminal_panel,
            evaluator_only=True,
        ), self.config.logging.video.terminal_hold_seconds

    def _render_canvas(
        self,
        sources: list[dict[str, Any]],
        panel: dict[str, Any],
        *,
        evaluator_only: bool,
    ) -> Image.Image:
        canvas = Image.new("RGB", CANVAS_SIZE, color=(18, 20, 24))
        draw = ImageDraw.Draw(canvas)
        font = _font(16)
        heading_font = _font(20)
        _draw_mosaic(canvas, draw, sources, self.episode_dir, font)
        draw.rectangle(
            (IMAGE_REGION_WIDTH, 0, CANVAS_SIZE[0], CANVAS_SIZE[1]),
            fill=(31, 35, 41),
        )
        y = 20
        if evaluator_only:
            draw.rectangle(
                (IMAGE_REGION_WIDTH + 16, y, CANVAS_SIZE[0] - 16, y + 38),
                fill=(148, 42, 42),
            )
            draw.text(
                (IMAGE_REGION_WIDTH + 28, y + 6),
                "EVALUATOR ONLY",
                fill=(255, 255, 255),
                font=heading_font,
            )
            y += 54
        for line in _panel_lines(panel):
            wrapped = textwrap.wrap(line, width=48, break_long_words=True) or [""]
            for part in wrapped:
                if y > CANVAS_SIZE[1] - 22:
                    return canvas
                draw.text(
                    (IMAGE_REGION_WIDTH + 20, y),
                    part,
                    fill=(232, 235, 239),
                    font=font,
                )
                y += 20
            y += 2
        return canvas

    def _render_output(
        self,
        target: Path,
        frames: Iterable[tuple[Image.Image, float]],
    ) -> dict[str, Any] | None:
        try:
            self._encode_video(target, frames)
        except Exception as exc:
            target.unlink(missing_ok=True)
            return _error_record(target.stem, exc)
        return None

    def _encode_video(
        self,
        target: Path,
        frames: Iterable[tuple[Image.Image, float]],
    ) -> None:
        fps = self.config.logging.video.fps
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
        frame_count = 0
        with tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
            except FileNotFoundError as exc:
                raise ReviewVideoError(f"FFmpeg executable not found: {self.ffmpeg_binary}") from exc
            assert process.stdin is not None
            try:
                for frame, duration in frames:
                    if frame.size != CANVAS_SIZE:
                        raise ReviewVideoError(f"Renderer produced invalid frame size: {frame.size}")
                    repeats = max(1, int(round(duration * fps)))
                    content = frame.tobytes()
                    for _ in range(repeats):
                        process.stdin.write(content)
                        frame_count += 1
                process.stdin.close()
                return_code = process.wait(timeout=120)
            except Exception:
                process.kill()
                process.wait()
                raise
            finally:
                if not process.stdin.closed:
                    process.stdin.close()
            stderr.seek(0)
            error_text = stderr.read().decode("utf-8", errors="replace").strip()
        if frame_count == 0:
            raise ReviewVideoError("Review video contains no frames")
        if return_code != 0:
            raise ReviewVideoError(
                f"FFmpeg exited with status {return_code}: {_sanitize_error_text(error_text)}"
            )
        if not target.is_file() or target.stat().st_size == 0:
            raise ReviewVideoError("FFmpeg did not produce a non-empty MP4")

    def _ffmpeg_version(self) -> str:
        try:
            result = subprocess.run(
                [self.ffmpeg_binary, "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError as exc:
            raise ReviewVideoError(f"FFmpeg executable not found: {self.ffmpeg_binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ReviewVideoError("FFmpeg version check timed out") from exc
        if result.returncode != 0:
            raise ReviewVideoError(
                f"FFmpeg version check failed: {_sanitize_error_text(result.stderr)}"
            )
        return result.stdout.splitlines()[0] if result.stdout else "unknown"

    def _artifact_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReviewVideoError(f"Review artifact path escapes episode directory: {relative_path!r}")
        root = self.episode_dir.resolve()
        target = (self.episode_dir / relative).resolve()
        if not target.is_relative_to(root):
            raise ReviewVideoError(f"Review artifact path escapes episode directory: {relative_path!r}")
        return target


def model_review_record(
    transition: StepTransition,
    episode_metadata: dict[str, Any],
) -> dict[str, Any]:
    trace = transition.agent_trace
    control_interface = episode_metadata.get("control_interface")
    public_control: dict[str, Any] = {}
    if isinstance(control_interface, dict):
        for key in ("mode", "active_dimensions", "gripper", "mapping"):
            if key in control_interface:
                public_control[key] = to_jsonable(control_interface[key])
    return {
        "task": str(episode_metadata.get("task_instruction", "")),
        "step": transition.step_id,
        "control": public_control,
        "raw_action": to_jsonable(transition.raw_action),
        "parsed_action": to_jsonable(transition.parsed_action),
        "reflection": transition.reflection,
        "action_status": _action_status(transition),
        "valid": transition.validation.is_valid,
        "validation_errors": [
            _sanitize_error_text(error) for error in transition.validation.errors
        ],
        "api": {
            "status": trace.status if trace is not None else "unavailable",
            "transport": trace.api if trace is not None else None,
            "error": (
                _sanitize_error_text(trace.error)
                if trace is not None and trace.error is not None
                else None
            ),
            "model": trace.model if trace is not None else None,
            "reasoning_effort": (
                trace.reasoning_effort if trace is not None else None
            ),
            "previous_response_id": (
                trace.previous_response_id if trace is not None else None
            ),
            "response_id": trace.response_id if trace is not None else None,
            "context_recovery_count": (
                trace.context_recovery_count if trace is not None else 0
            ),
            "latency_ms": trace.latency_ms if trace is not None else None,
            "attempts": trace.attempts if trace is not None else 0,
            "input_tokens": trace.input_tokens if trace is not None else None,
            "output_tokens": trace.output_tokens if trace is not None else None,
            "cached_input_tokens": trace.cached_input_tokens if trace is not None else None,
            "cache_hit_ratio": trace.cache_hit_ratio if trace is not None else None,
            "cache_metrics_status": (
                trace.cache_metrics_status if trace is not None else "unavailable"
            ),
        },
    }


def evaluator_debug_record(transition: StepTransition) -> dict[str, Any]:
    state = transition.next_observation.state
    info = transition.info
    return {
        "reward": _first_value(info, state, keys=("reward",)),
        "distance": _first_value(info, state, keys=_ORACLE_DISTANCE_KEYS),
        "success": _first_value(
            info,
            state,
            keys=("success", "goal_termination", "task_success"),
        ),
        "target_position": _first_value(info, state, keys=_ORACLE_TARGET_KEYS),
        "eef_position": _first_value(info, state, keys=_ORACLE_EEF_KEYS),
        "simulator_action": to_jsonable(transition.sim_action),
    }


def evaluator_terminal_record(
    observation: Observation,
    result: EpisodeResult,
) -> dict[str, Any]:
    state = observation.state
    return {
        "reward": _first_value({}, state, keys=("reward",)),
        "distance": _first_value({}, state, keys=_ORACLE_DISTANCE_KEYS),
        "success": result.success,
        "target_position": _first_value({}, state, keys=_ORACLE_TARGET_KEYS),
        "eef_position": _first_value({}, state, keys=_ORACLE_EEF_KEYS),
        "simulator_action": None,
    }


def _draw_mosaic(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    sources: list[dict[str, Any]],
    episode_dir: Path,
    font: ImageFont.ImageFont,
) -> None:
    columns = max(1, math.ceil(math.sqrt(len(sources))))
    rows = max(1, math.ceil(len(sources) / columns))
    cell_width = IMAGE_REGION_WIDTH // columns
    cell_height = CANVAS_SIZE[1] // rows
    label_height = 30
    for index, source in enumerate(sources):
        column = index % columns
        row = index // columns
        x0 = column * cell_width
        y0 = row * cell_height
        path = episode_dir / source["artifact"]
        with Image.open(path) as opened:
            frame = opened.convert("RGB")
        frame.thumbnail(
            (max(1, cell_width - 16), max(1, cell_height - label_height - 16)),
            Image.Resampling.LANCZOS,
        )
        x = x0 + (cell_width - frame.width) // 2
        y = y0 + label_height + (cell_height - label_height - frame.height) // 2
        canvas.paste(frame, (x, y))
        draw.text(
            (x0 + 10, y0 + 5),
            str(source["camera"]),
            fill=(238, 241, 245),
            font=font,
        )


def _panel_lines(panel: dict[str, Any]) -> list[str]:
    if "api" in panel and "raw_action" in panel:
        return _model_panel_lines(panel)
    lines: list[str] = []
    for key, value in panel.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for nested_key, nested_value in value.items():
                lines.append(f"  {nested_key}: {_compact(nested_value)}")
        else:
            lines.append(f"{key}: {_compact(value)}")
    return lines


def _model_panel_lines(panel: dict[str, Any]) -> list[str]:
    lines = [
        f"task: {_compact(panel.get('task'))}",
        f"step: {_compact(panel.get('step'))}",
    ]
    control = panel.get("control")
    if isinstance(control, dict):
        lines.append(
            "control: "
            f"mode={_compact(control.get('mode'))} "
            f"dims={_compact(control.get('active_dimensions'))} "
            f"gripper={_compact(control.get('gripper'))}"
        )
        if "mapping" in control:
            lines.append(f"mapping: {_mapping_summary(control['mapping'])}")
    lines.extend(
        [
            f"raw action: {_compact(panel.get('raw_action'))}",
            f"parsed action: {_compact(panel.get('parsed_action'))}",
            "action result: "
            f"{_compact(panel.get('action_status'))}; valid={_compact(panel.get('valid'))}",
        ]
    )
    errors = panel.get("validation_errors")
    if errors:
        lines.append(f"validation errors: {_compact(errors)}")
    api = panel.get("api")
    if isinstance(api, dict):
        lines.extend(
            [
                "api: "
                f"status={_compact(api.get('status'))} "
                f"model={_compact(api.get('model'))} "
                f"reasoning={_compact(api.get('reasoning_effort'))}",
                "api timing: "
                f"latency_ms={_compact(api.get('latency_ms'))} "
                f"attempts={_compact(api.get('attempts'))} "
                f"context_recoveries={_compact(api.get('context_recovery_count'))}",
                "tokens: "
                f"input={_compact(api.get('input_tokens'))} "
                f"output={_compact(api.get('output_tokens'))} "
                f"cached={_compact(api.get('cached_input_tokens'))}",
                "cache: "
                f"status={_compact(api.get('cache_metrics_status'))} "
                f"hit_ratio={_compact(api.get('cache_hit_ratio'))}",
            ]
        )
        if api.get("error"):
            lines.append(f"api error: {_compact(api['error'])}")
    evaluator = panel.get("evaluator")
    if isinstance(evaluator, dict):
        lines.append("evaluator:")
        for key, value in evaluator.items():
            lines.append(f"  {key}: {_compact(value)}")
    return lines


def _mapping_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return _compact(value)
    channels = value.get("channels")
    signs = value.get("signs")
    if not isinstance(channels, dict):
        return _compact(value)
    entries = []
    for channel, dimension in channels.items():
        suffix = ""
        if isinstance(signs, dict) and channel in signs:
            suffix = f"({float(signs[channel]):+g})"
        entries.append(f"{channel}->{dimension}{suffix}")
    return ", ".join(entries)


def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(to_jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _action_status(transition: StepTransition) -> str:
    if transition.agent_trace is not None and transition.agent_trace.status == "error":
        return "api_error_noop"
    if not transition.validation.is_valid:
        return "invalid_action_noop"
    return "valid"


def _first_value(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> Any:
    for source in (first, second):
        for key in keys:
            if key in source:
                return to_jsonable(source[key])
    return None


def _output_record(path: Path, error: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "status": "failed" if error is not None else "completed",
        "path": path.name if error is None else None,
        "error": error,
    }


def _error_record(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": _sanitize_error_text(str(exc)),
    }


def _sanitize_error_text(value: str) -> str:
    sanitized = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", value)
    sanitized = re.sub(r"(?i)data:image/[^;\s]+;base64,\S+", "[image data omitted]", sanitized)
    sanitized = re.sub(
        r"(?i)(authorization\s*[:=]\s*)\S+",
        r"\1[REDACTED]",
        sanitized,
    )
    return sanitized[:2000]


def _relative_to_episode(path: Path, episode_dir: Path) -> str:
    return path.relative_to(episode_dir).as_posix()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
