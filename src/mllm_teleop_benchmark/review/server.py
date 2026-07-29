from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mllm_teleop_benchmark.review.store import ReviewCatalogStore, ReviewDataError


STATIC_DIR = Path(__file__).with_name("static")


class ReviewRequestHandler(BaseHTTPRequestHandler):
    store: ReviewCatalogStore

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
                return
            if parsed.path.startswith("/static/"):
                name = parsed.path.removeprefix("/static/")
                if "/" in name or name not in {"app.js", "styles.css"}:
                    self._send_error(HTTPStatus.NOT_FOUND, "Static asset not found")
                    return
                self._send_file(STATIC_DIR / name)
                return
            if parsed.path == "/api/index":
                self._send_json(self.store.index())
                return
            if parsed.path.startswith("/api/runs/"):
                self._handle_run_api(parsed.path)
                return
            if parsed.path.startswith("/api/episodes/"):
                self._handle_episode_api(parsed.path)
                return
            if parsed.path.startswith("/artifacts/runs/"):
                self._handle_run_artifact(parsed.path)
                return
            if parsed.path.startswith("/artifacts/"):
                self._handle_artifact(parsed.path)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Route not found")
        except ReviewDataError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self._send_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Review server error: {type(exc).__name__}",
            )

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _handle_episode_api(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) == 4 and parts[:2] == ["api", "episodes"] and parts[3] == "conversation":
            self._send_json(
                self.store.conversation(self.store.default_run_id, parts[2])
            )
            return
        if (
            len(parts) == 5
            and parts[:2] == ["api", "episodes"]
            and parts[3] == "steps"
        ):
            try:
                step_id = int(parts[4])
            except ValueError as exc:
                raise ReviewDataError(f"Invalid step id: {parts[4]}") from exc
            self._send_json(
                self.store.step(self.store.default_run_id, parts[2], step_id)
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Episode API route not found")

    def _handle_run_api(self, path: str) -> None:
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) == 3 and parts[:2] == ["api", "runs"]:
            self._send_json(self.store.run_index(parts[2]))
            return
        if (
            len(parts) == 6
            and parts[:2] == ["api", "runs"]
            and parts[3] == "episodes"
            and parts[5] == "conversation"
        ):
            self._send_json(self.store.conversation(parts[2], parts[4]))
            return
        if (
            len(parts) == 7
            and parts[:2] == ["api", "runs"]
            and parts[3] == "episodes"
            and parts[5] == "steps"
        ):
            try:
                step_id = int(parts[6])
            except ValueError as exc:
                raise ReviewDataError(f"Invalid step id: {parts[6]}") from exc
            self._send_json(self.store.step(parts[2], parts[4], step_id))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Run API route not found")

    def _handle_artifact(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 4:
            self._send_error(HTTPStatus.NOT_FOUND, "Artifact route not found")
            return
        episode_id = unquote(parts[2])
        relative_path = unquote("/".join(parts[3:]))
        self._send_file(
            self.store.artifact(
                self.store.default_run_id,
                episode_id,
                relative_path,
            )
        )

    def _handle_run_artifact(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 6:
            self._send_error(HTTPStatus.NOT_FOUND, "Run artifact route not found")
            return
        run_id = unquote(parts[3])
        episode_id = unquote(parts[4])
        relative_path = unquote("/".join(parts[5:]))
        self._send_file(self.store.artifact(run_id, episode_id, relative_path))

    def _send_json(self, value: object) -> None:
        content = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = path.read_bytes()
        content_type, _encoding = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        content = json.dumps({"error": message}, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _security_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )


def build_server(
    store: ReviewCatalogStore,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    handler = type(
        "BoundReviewRequestHandler",
        (ReviewRequestHandler,),
        {"store": store},
    )
    return ThreadingHTTPServer((host, port), handler)


def serve_review(
    path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = build_server(ReviewCatalogStore(path), host, port)
    address, actual_port = server.server_address[:2]
    print(f"Review UI: http://{address}:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
