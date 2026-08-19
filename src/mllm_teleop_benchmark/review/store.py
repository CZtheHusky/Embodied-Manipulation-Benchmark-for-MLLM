from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


class ReviewDataError(ValueError):
    pass


class EpisodeReviewStore:
    def __init__(
        self,
        path: str | Path,
        *,
        artifact_prefix: str = "/artifacts",
    ) -> None:
        self.root = Path(path).expanduser().resolve()
        self.artifact_prefix = artifact_prefix.rstrip("/")
        self._episodes = self._discover_episodes(self.root)
        self._trajectory_cache: dict[str, dict[int, dict[str, Any]]] = {}
        self._allowed_artifacts: dict[str, set[str]] = {}

    def index(self) -> dict[str, Any]:
        episodes = []
        for episode_id, episode_dir in self._episodes.items():
            summary = self._read_optional_json(episode_dir / "summary.json") or {}
            steps = self._step_ids(episode_dir)
            trajectory = self._trajectory(episode_id)
            step_entries = []
            for step_id in steps:
                transition = trajectory.get(step_id, {})
                trace = transition.get("agent_trace")
                validation = transition.get("validation")
                step_entries.append(
                    {
                        "id": step_id,
                        "status": (
                            trace.get("status")
                            if isinstance(trace, dict)
                            else (
                                "error"
                                if (
                                    episode_dir
                                    / "model_calls"
                                    / f"step_{step_id:04d}"
                                    / "error.json"
                                ).is_file()
                                else (
                                    "ok"
                                    if (
                                        episode_dir
                                        / "model_calls"
                                        / f"step_{step_id:04d}"
                                        / "response.json"
                                    ).is_file()
                                    else "pending"
                                )
                            )
                        ),
                        "valid": (
                            validation.get("is_valid")
                            if isinstance(validation, dict)
                            else None
                        ),
                    }
                )
            episodes.append(
                {
                    "id": episode_id,
                    "episode_id": summary.get(
                        "episode_id",
                        _episode_number(episode_id),
                    ),
                    "success": bool(summary.get("success", False)),
                    "steps": int(summary.get("steps", len(steps))),
                    "termination_reason": summary.get(
                        "termination_reason",
                        "incomplete",
                    ),
                    "prompt_version": self._prompt_version(episode_dir, steps),
                    "step_ids": steps,
                    "step_entries": step_entries,
                    "video_status": (
                        summary.get("review_video", {}).get("status")
                        if isinstance(summary.get("review_video"), dict)
                        else None
                    ),
                }
            )
        return {
            "name": self.root.name,
            "episodes": episodes,
        }

    def step(self, episode_id: str, step_id: int) -> dict[str, Any]:
        episode_dir = self._episode_dir(episode_id)
        artifact_dir = episode_dir / "model_calls" / f"step_{step_id:04d}"
        request_wrapper = self._read_json(artifact_dir / "request.json")
        request = request_wrapper.get("request")
        display_messages = _display_messages(request)
        if not isinstance(request, dict) or display_messages is None:
            raise ReviewDataError(f"Invalid request artifact for {episode_id} step {step_id}")
        images = self._images(episode_id, episode_dir, request_wrapper)
        response = self._read_optional_json(artifact_dir / "response.json")
        error = self._read_optional_json(artifact_dir / "error.json")
        transition = self._trajectory(episode_id).get(step_id, {})
        trace = transition.get("agent_trace") if isinstance(transition, dict) else None
        return {
            "episode": episode_id,
            "step_id": step_id,
            "prompt_version": request_wrapper.get("prompt_version"),
            "common_prefix_bytes": request_wrapper.get("common_prefix_bytes", 0),
            "api": request_wrapper.get("api", _request_api(request)),
            "request": request,
            "display_messages": display_messages,
            "current_observation": _current_observation(
                request,
                request_wrapper,
                display_messages,
            ),
            "images": images,
            "response": response,
            "response_content": _response_content(response),
            "error": error,
            "public_trace": _public_trace(trace),
            "action": {
                "raw": transition.get("raw_action") if isinstance(transition, dict) else None,
                "parsed": transition.get("parsed_action") if isinstance(transition, dict) else None,
                "reflection": transition.get("reflection") if isinstance(transition, dict) else None,
                "magnifier_update": (
                    transition.get("magnifier_update")
                    if isinstance(transition, dict)
                    else None
                ),
                "magnifier_result": (
                    transition.get("magnifier_result")
                    if isinstance(transition, dict)
                    else None
                ),
                "validation": transition.get("validation") if isinstance(transition, dict) else None,
            },
        }

    def conversation(self, episode_id: str) -> dict[str, Any]:
        episode_dir = self._episode_dir(episode_id)
        turns = []
        static_messages: list[dict[str, Any]] = []
        for step_id in self._step_ids(episode_dir):
            step = self.step(episode_id, step_id)
            if not static_messages:
                static_messages = _static_display_messages(step)
            turns.append(
                {
                    "step_id": step_id,
                    "observation": step["current_observation"],
                    "images": [
                        image for image in step["images"] if image.get("step_id") == step_id
                    ],
                    "response_content": step["response_content"],
                    "response": step["response"],
                    "error": step["error"],
                    "public_trace": step["public_trace"],
                    "action": step["action"],
                }
            )
        return {
            "episode": episode_id,
            "prompt_version": (
                self._prompt_version(episode_dir, self._step_ids(episode_dir))
            ),
            "static_messages": static_messages,
            "turns": turns,
        }

    def artifact(self, episode_id: str, relative_path: str) -> Path:
        episode_dir = self._episode_dir(episode_id)
        normalized = Path(relative_path)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ReviewDataError("Artifact path must stay within the episode directory")
        allowed = self._artifact_allowlist(episode_id, episode_dir)
        posix_path = normalized.as_posix()
        if posix_path not in allowed:
            raise ReviewDataError("Artifact is not a model-visible request image")
        target = (episode_dir / normalized).resolve()
        if not target.is_relative_to(episode_dir):
            raise ReviewDataError("Artifact path escapes the episode directory")
        if not target.is_file():
            raise ReviewDataError(f"Artifact does not exist: {relative_path}")
        return target

    def _discover_episodes(self, root: Path) -> dict[str, Path]:
        if not root.is_dir():
            raise ReviewDataError(f"Review path does not exist: {root}")
        if _episode_has_requests(root):
            candidates = [root]
        else:
            candidates = sorted(
                path
                for path in root.iterdir()
                if path.is_dir()
                and path.name.startswith("episode_")
                and _episode_has_requests(path)
            )
        if not candidates:
            raise ReviewDataError(
                "Review path must be an episode directory or a run containing "
                "episode_* model request artifacts"
            )
        return {path.name: path.resolve() for path in candidates}

    def _episode_dir(self, episode_id: str) -> Path:
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise ReviewDataError(f"Unknown episode: {episode_id}") from exc

    def _step_ids(self, episode_dir: Path) -> list[int]:
        step_ids = []
        model_calls = episode_dir / "model_calls"
        if not model_calls.is_dir():
            return step_ids
        for request_path in model_calls.glob("step_*/request.json"):
            try:
                step_ids.append(int(request_path.parent.name.removeprefix("step_")))
            except ValueError:
                continue
        return sorted(set(step_ids))

    def _prompt_version(self, episode_dir: Path, step_ids: list[int]) -> str | None:
        if not step_ids:
            return None
        request = self._read_json(
            episode_dir / "model_calls" / f"step_{step_ids[0]:04d}" / "request.json"
        )
        value = request.get("prompt_version")
        return str(value) if value is not None else None

    def _images(
        self,
        episode_id: str,
        episode_dir: Path,
        request_wrapper: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records = request_wrapper.get("images")
        if not isinstance(records, list):
            return []
        images = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("artifact"), str):
                continue
            artifact = str(record["artifact"])
            images.append(
                {
                    "step_id": record.get("step_id"),
                    "camera": record.get("camera"),
                    "width": record.get("width"),
                    "height": record.get("height"),
                    "sha256": record.get("sha256"),
                    "kind": record.get("kind", "rgb"),
                    "source_rgb_key": record.get("source_rgb_key"),
                    "bbox_xyxy": record.get("bbox_xyxy"),
                    "artifact": artifact,
                    "url": (
                        f"{self.artifact_prefix}/{quote(episode_id, safe='')}/"
                        f"{quote(artifact, safe='/')}"
                    ),
                }
            )
        return images

    def _trajectory(self, episode_id: str) -> dict[int, dict[str, Any]]:
        cached = self._trajectory_cache.get(episode_id)
        if cached is not None:
            return cached
        path = self._episode_dir(episode_id) / "trajectory.jsonl"
        transitions: dict[int, dict[str, Any]] = {}
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if isinstance(value, dict) and isinstance(value.get("step_id"), int):
                        transitions[int(value["step_id"])] = value
        self._trajectory_cache[episode_id] = transitions
        return transitions

    def _artifact_allowlist(self, episode_id: str, episode_dir: Path) -> set[str]:
        cached = self._allowed_artifacts.get(episode_id)
        if cached is not None:
            return cached
        allowed: set[str] = set()
        for step_id in self._step_ids(episode_dir):
            request = self._read_json(
                episode_dir / "model_calls" / f"step_{step_id:04d}" / "request.json"
            )
            records = request.get("images")
            if not isinstance(records, list):
                continue
            for record in records:
                if isinstance(record, dict) and isinstance(record.get("artifact"), str):
                    allowed.add(Path(record["artifact"]).as_posix())
        self._allowed_artifacts[episode_id] = allowed
        return allowed

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ReviewDataError(f"Missing review artifact: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ReviewDataError(f"Invalid JSON review artifact: {path}") from exc
        if not isinstance(value, dict):
            raise ReviewDataError(f"Expected a JSON object: {path}")
        return value

    @classmethod
    def _read_optional_json(cls, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        return cls._read_json(path)


class ReviewCatalogStore:
    def __init__(self, path: str | Path) -> None:
        self.root = Path(path).expanduser().resolve()
        self._runs: dict[str, Path] = {}
        self._stores: dict[str, EpisodeReviewStore] = {}
        self.default_run_id = ""
        self._refresh()

    def index(self) -> dict[str, Any]:
        self._refresh()
        groups: dict[str, dict[str, Any]] = {}
        for run_id, run_dir in self._runs.items():
            group_id = self._group_id(run_id)
            group = groups.setdefault(
                group_id,
                {
                    "id": group_id,
                    "name": group_id if group_id != "." else self.root.name,
                    "runs": [],
                },
            )
            run_summary = EpisodeReviewStore._read_optional_json(
                run_dir / "run_summary.json"
            )
            run_manifest = EpisodeReviewStore._read_optional_json(
                run_dir / "run_manifest.json"
            )
            group["runs"].append(
                {
                    "id": run_id,
                    "name": run_dir.name,
                    "relative_path": run_id,
                    "episode_count": len(self._reviewable_episodes(run_dir)),
                    "aggregate_metrics": (
                        run_summary.get("aggregate_metrics")
                        if isinstance(run_summary, dict)
                        else None
                    ),
                    "model": _manifest_value(
                        run_manifest,
                        "model",
                        "requested_model",
                    ),
                    "reasoning_effort": _manifest_value(
                        run_manifest,
                        "model",
                        "reasoning_effort",
                    ),
                    "api": _manifest_value(run_manifest, "model", "api"),
                }
            )
        ordered_groups = sorted(
            groups.values(),
            key=lambda group: (_catalog_group_rank(str(group["id"])), str(group["id"])),
        )
        for group in ordered_groups:
            group["runs"].sort(key=lambda run: str(run["name"]), reverse=True)
        return {
            "name": self.root.name,
            "root": str(self.root),
            "default_run_id": self.default_run_id,
            "default_group_id": self._group_id(self.default_run_id),
            "groups": ordered_groups,
        }

    def run_index(self, run_id: str) -> dict[str, Any]:
        value = self._store(run_id).index()
        return {
            **value,
            "run_id": run_id,
            "relative_path": run_id,
            "group_id": self._group_id(run_id),
        }

    def step(self, run_id: str, episode_id: str, step_id: int) -> dict[str, Any]:
        return self._store(run_id).step(episode_id, step_id)

    def conversation(self, run_id: str, episode_id: str) -> dict[str, Any]:
        return self._store(run_id).conversation(episode_id)

    def artifact(self, run_id: str, episode_id: str, relative_path: str) -> Path:
        return self._store(run_id).artifact(episode_id, relative_path)

    def _store(self, run_id: str) -> EpisodeReviewStore:
        if run_id not in self._runs or not self._runs[run_id].is_dir():
            self._refresh()
        cached = self._stores.get(run_id)
        if cached is not None:
            return cached
        try:
            run_dir = self._runs[run_id]
        except KeyError as exc:
            raise ReviewDataError(f"Unknown review run: {run_id}") from exc
        store = EpisodeReviewStore(
            run_dir,
            artifact_prefix=f"/artifacts/runs/{quote(run_id, safe='')}",
        )
        self._stores[run_id] = store
        return store

    def _refresh(self) -> None:
        runs = self._discover_runs(self.root)
        self._runs = runs
        self._stores.clear()
        self.default_run_id = self._default_run_id()

    def _discover_runs(self, root: Path) -> dict[str, Path]:
        if not root.is_dir():
            raise ReviewDataError(f"Review path does not exist: {root}")
        if _episode_has_requests(root):
            candidates = {root}
        elif self._reviewable_episodes(root):
            candidates = {root}
        else:
            candidates = {
                model_calls.parent.parent.resolve()
                for model_calls in root.rglob("model_calls")
                if model_calls.is_dir()
                and "_trash" not in model_calls.relative_to(root).parts
                and model_calls.parent.name.startswith("episode_")
                and self._has_model_requests(model_calls.parent)
                and model_calls.parent.resolve().is_relative_to(root)
            }
        if not candidates:
            raise ReviewDataError(
                "Review path contains no episodes with model request artifacts"
            )
        return {
            self._relative_run_id(path): path
            for path in sorted(candidates, key=lambda value: value.as_posix())
        }

    def _default_run_id(self) -> str:
        normal = [
            (run_id, run_dir)
            for run_id, run_dir in self._runs.items()
            if _catalog_group_rank(self._group_id(run_id)) == 0
        ]
        candidates = normal or list(self._runs.items())
        return max(candidates, key=lambda item: item[1].stat().st_mtime_ns)[0]

    def _relative_run_id(self, run_dir: Path) -> str:
        if run_dir == self.root:
            return "."
        return run_dir.relative_to(self.root).as_posix()

    @staticmethod
    def _group_id(run_id: str) -> str:
        if run_id == ".":
            return "."
        parent = Path(run_id).parent.as_posix()
        return parent if parent else "."

    @classmethod
    def _reviewable_episodes(cls, run_dir: Path) -> list[Path]:
        if _episode_has_requests(run_dir):
            return [run_dir] if cls._has_model_requests(run_dir) else []
        return sorted(
            path
            for path in run_dir.iterdir()
            if path.is_dir()
            and path.name.startswith("episode_")
            and cls._has_model_requests(path)
        )

    @staticmethod
    def _has_model_requests(episode_dir: Path) -> bool:
        return _episode_has_requests(episode_dir)


def _manifest_value(
    manifest: dict[str, Any] | None,
    section: str,
    key: str,
) -> Any:
    if not isinstance(manifest, dict):
        return None
    value = manifest.get(section)
    return value.get(key) if isinstance(value, dict) else None


def _catalog_group_rank(group_id: str) -> int:
    parts = set(Path(group_id).parts)
    if "repairs" in parts:
        return 1
    if "archive" in parts:
        return 2
    if "invalid" in parts:
        return 3
    if "interrupted" in parts:
        return 4
    return 0


def _episode_has_requests(episode_dir: Path) -> bool:
    model_calls = episode_dir / "model_calls"
    return model_calls.is_dir() and next(
        model_calls.glob("step_*/request.json"),
        None,
    ) is not None


def _episode_number(episode_id: str) -> int | None:
    try:
        return int(episode_id.removeprefix("episode_"))
    except ValueError:
        return None


def _response_content(response: dict[str, Any] | None) -> Any:
    if not isinstance(response, dict):
        return None
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    output = response.get("output")
    if isinstance(output, list):
        parts = [
            content.get("text")
            for item in output
            if isinstance(item, dict) and item.get("type") == "message"
            for content in (
                item.get("content") if isinstance(item.get("content"), list) else ()
            )
            if isinstance(content, dict)
            and content.get("type") == "output_text"
            and isinstance(content.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    return message.get("content") if isinstance(message, dict) else None


def _public_trace(trace: Any) -> dict[str, Any] | None:
    if not isinstance(trace, dict):
        return None
    allowed = (
        "actor_type",
        "status",
        "provider",
        "api",
        "model",
        "reasoning_effort",
        "request_id",
        "previous_response_id",
        "response_id",
        "context_recovery_count",
        "latency_ms",
        "attempts",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_hit_ratio",
        "cache_metrics_status",
        "common_prefix_bytes",
        "request_bytes",
        "image_count",
        "error",
        "prompt_version",
        "artifact_dir",
    )
    return {key: trace.get(key) for key in allowed}


def _request_api(request: dict[str, Any]) -> str:
    return "responses" if isinstance(request.get("input"), list) else "chat_completions"


def _display_messages(request: Any) -> list[dict[str, Any]] | None:
    if not isinstance(request, dict):
        return None
    messages = request.get("messages")
    if isinstance(messages, list):
        return messages
    input_items = request.get("input")
    instructions = request.get("instructions")
    if not isinstance(input_items, list) or not isinstance(instructions, str):
        return None
    display: list[dict[str, Any]] = [{"role": "system", "content": instructions}]
    display.extend(
        _normalize_responses_message(item)
        for item in input_items
        if isinstance(item, dict)
    )
    return display


def _normalize_responses_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return dict(message)
    normalized = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append(item)
        elif item.get("type") == "input_text":
            normalized.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "input_image":
            normalized.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item.get("image_url", "")},
                    "artifact": item.get("artifact"),
                }
            )
        else:
            normalized.append(dict(item))
    return {**message, "content": normalized}


def _current_observation(
    request: dict[str, Any],
    request_wrapper: dict[str, Any],
    display_messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if _request_api(request) == "chat_completions":
        return display_messages[-1] if display_messages else None
    index = request_wrapper.get("current_input_index")
    if not isinstance(index, int):
        return None
    display_index = index + 1
    if display_index < 1 or display_index >= len(display_messages):
        return None
    return display_messages[display_index]


def _static_display_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    messages = step["display_messages"]
    if step["api"] == "responses":
        return messages[:1]
    return messages[:2]
