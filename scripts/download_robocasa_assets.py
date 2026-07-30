#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable
from zipfile import ZipFile


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
    extracted: list[str] = []
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
        target.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {filename} into {target}")
        _extract_zip(archive, target)
        extracted.append(name)

    marker = {
        "repository": HF_REPO,
        "revision": HF_REVISION,
        "asset_groups": extracted,
    }
    (assets_root / MARKER_NAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_zip(archive: Path, target: Path) -> None:
    target_resolved = target.resolve()
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target_resolved):
                raise ValueError(
                    f"Archive member escapes extraction directory: {member.filename}"
                )
        zip_file.extractall(target)


def _installed_assets_root() -> Path:
    import robocasa

    return Path(robocasa.__path__[0]) / "models" / "assets"


if __name__ == "__main__":
    main()
