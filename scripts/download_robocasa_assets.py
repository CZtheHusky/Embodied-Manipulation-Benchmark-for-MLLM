#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Callable

from mllm_teleop_benchmark.utils.pinned_zip import install_zip_exact


HF_REPO = "robocasa/robocasa-assets"
HF_REVISION = "1b92c3d02ca4354984fec961357db0bff7b32166"
ASSET_ARCHIVES = {
    "textures": ("textures.zip", Path(".")),
    "objaverse": ("objaverse.zip", Path("objects")),
}
MARKER_NAME = ".robocasa-assets-revision.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned RoboCasa textures and Objaverse assets."
    )
    parser.add_argument(
        "--type",
        choices=(*ASSET_ARCHIVES, "all"),
        default="all",
        help="Asset group to download.",
    )
    args = parser.parse_args()
    selected = tuple(ASSET_ARCHIVES) if args.type == "all" else (args.type,)
    download_assets(selected, assets_root=_installed_assets_root())


def download_assets(
    selected: tuple[str, ...],
    *,
    assets_root: Path,
    download_fn: Callable[..., str] | None = None,
) -> None:
    unknown = sorted(set(selected) - set(ASSET_ARCHIVES))
    if unknown:
        raise ValueError(f"Unknown RoboCasa asset groups: {unknown}")
    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download

    assets_root.mkdir(parents=True, exist_ok=True)
    marker_path = assets_root / MARKER_NAME
    previous_archives = _load_previous_archives(marker_path)
    installed: dict[str, dict[str, object]] = {}
    for name in selected:
        filename, relative_target = ASSET_ARCHIVES[name]
        print(f"Downloading {filename} from {HF_REPO}@{HF_REVISION}")
        archive = Path(
            download_fn(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=filename,
                revision=HF_REVISION,
            )
        )
        target = assets_root / relative_target
        print(f"Extracting {filename} into {target}")
        result = install_zip_exact(
            archive,
            target,
            previous_files=_previous_files(
                previous_archives,
                filename,
            ),
        )
        installed[filename] = {
            "sha256": result.sha256,
            "roots": list(result.roots),
            "file_count": result.file_count,
            "files": list(result.files),
        }

    marker = {
        "repository": HF_REPO,
        "revision": HF_REVISION,
        "asset_groups": list(selected),
        "archives": installed,
    }
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_previous_archives(
    marker_path: Path,
) -> Mapping[str, object]:
    if not marker_path.is_file():
        return {}
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    archives = marker.get("archives") if isinstance(marker, Mapping) else None
    if not isinstance(archives, Mapping):
        raise ValueError(
            f"Invalid RoboCasa asset marker: {marker_path}"
        )
    return archives


def _previous_files(
    archives: Mapping[str, object],
    filename: str,
) -> tuple[str, ...]:
    record = archives.get(filename)
    if not isinstance(record, Mapping):
        return ()
    files = record.get("files", ())
    if not isinstance(files, list) or not all(
        isinstance(value, str) for value in files
    ):
        raise ValueError(
            f"Invalid previous file manifest for {filename}"
        )
    return tuple(files)


def _installed_assets_root() -> Path:
    import robocasa

    return Path(robocasa.__path__[0]) / "models" / "assets"


if __name__ == "__main__":
    main()
