#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from mllm_teleop_benchmark.agents.model_policy import sanitize_provider_response


def sanitize_artifacts(root: Path, *, write: bool) -> dict[str, Any]:
    root = root.resolve()
    # rglob() on a missing path returns no files and would falsely report a successful scan.
    if not root.is_dir():
        raise ValueError(f"Artifact root is not a directory: {root}")

    changed: list[dict[str, str]] = []
    scanned = 0
    for path in sorted(root.rglob("response.json")):
        scanned += 1
        original = path.read_bytes()
        payload = json.loads(original)
        sanitized = sanitize_provider_response(payload)
        updated = (
            json.dumps(sanitized, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        if updated == original:
            continue
        changed.append(
            {
                "path": str(path.relative_to(root)),
                "sha256_before": hashlib.sha256(original).hexdigest(),
                "sha256_after": hashlib.sha256(updated).hexdigest(),
            }
        )
        if write:
            temporary = path.with_suffix(".json.sanitizing")
            temporary.write_bytes(updated)
            temporary.replace(path)

    return {
        "schema_version": 1,
        "redaction": "hidden_reasoning_v1",
        "root": str(root),
        "scanned_response_artifacts": scanned,
        "changed_response_artifacts": len(changed),
        "files": changed,
    }


def _write_manifest(root: Path, report: dict[str, Any]) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    manifest_dir = root / "artifact_sanitization"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{timestamp}-hidden-reasoning-v1.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redact hidden reasoning from persisted model response artifacts."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = sanitize_artifacts(args.root, write=not args.dry_run)
    if not args.dry_run:
        report["manifest"] = str(_write_manifest(args.root.resolve(), report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
