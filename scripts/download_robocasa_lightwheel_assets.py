#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Callable

from mllm_teleop_benchmark.utils.pinned_zip import (
    ZipInstallRequest,
    install_zip_batch_exact,
    write_json_atomic,
)

HF_REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
HF_REVISION = "420a04af939c34873e6839a586b70844baf28aab"
MARKER_NAME = ".robocasa-lightwheel-revision.json"
MARKER_SCHEMA_VERSION = 1

FIXTURE_ZIPS = (
    "blenders.zip",
    "cabinets.zip",
    "coffee_machines.zip",
    "dishwashers.zip",
    "electric_kettles.zip",
    "fridges.zip",
    "handles.zip",
    "hoods.zip",
    "microwaves.zip",
    "ovens.zip",
    "sinks.zip",
    "stand_mixers.zip",
    "stoves.zip",
    "stovetops.zip",
    "toaster_ovens.zip",
    "toasters.zip",
    "windows.zip",
)

OBJECT_ZIPS = (
    "aluminum_foil.zip",
    "basket.zip",
    "blender_jug.zip",
    "cheese_grater.zip",
    "chicken_drumstick.zip",
    "cinnamon.zip",
    "colander.zip",
    "cookie_dough_ball.zip",
    "cream_cheese_stick.zip",
    "digital_scale.zip",
    "dish_brush.zip",
    "dish_rack.zip",
    "flour_bag.zip",
    "flower_vase.zip",
    "fruit_bowl.zip",
    "glass_cup.zip",
    "honey_bottle.zip",
    "hotdog_bun.zip",
    "ice_cube.zip",
    "ice_cube_tray.zip",
    "jar.zip",
    "juice.zip",
    "kebab_skewer.zip",
    "kettle.zip",
    "knife_block.zip",
    "lemon_wedge.zip",
    "lettuce.zip",
    "marshmallow.zip",
    "mayonnaise.zip",
    "measuring_cup.zip",
    "mug_tree.zip",
    "mustard.zip",
    "oil_and_vinegar_bottle.zip",
    "oven_tray.zip",
    "pancake.zip",
    "paper_towel_holder.zip",
    "paprika.zip",
    "peeler.zip",
    "pickle_slice.zip",
    "pitcher.zip",
    "pizza.zip",
    "pizza_cutter.zip",
    "placemat.zip",
    "plant.zip",
    "pot.zip",
    "reamer.zip",
    "salt_and_pepper_shaker.zip",
    "sandwich_bread.zip",
    "saucepan.zip",
    "shrimp.zip",
    "soap_dispenser.zip",
    "spray.zip",
    "stool.zip",
    "strainer.zip",
    "straw.zip",
    "sugar_cube.zip",
    "syrup_bottle.zip",
    "tiered_basket.zip",
    "tiered_shelf.zip",
    "tomato_slice.zip",
    "tongs.zip",
    "tray.zip",
    "tupperware.zip",
    "turkey_slice.zip",
    "turmeric.zip",
    "utensil_rack.zip",
    "utensil_set.zip",
    "whisk.zip",
    "wooden_spoon.zip",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned Lightwheel RoboCasa assets."
    )
    parser.add_argument(
        "--type",
        choices=("fixtures", "objects", "all"),
        default="all",
        help="Asset group to download.",
    )
    args = parser.parse_args()
    selected = ("fixtures", "objects") if args.type == "all" else (args.type,)
    download_assets(selected, assets_root=_installed_assets_root())


def download_assets(
    selected: tuple[str, ...],
    *,
    assets_root: Path,
    download_fn: Callable[..., str] | None = None,
) -> None:
    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download

    marker_path = assets_root / MARKER_NAME
    previous_archives = _load_previous_archives(marker_path)
    groups = {
        "fixtures": (
            "fixtures_lightwheel",
            FIXTURE_ZIPS,
            assets_root / "fixtures",
        ),
        "objects": (
            "objects_lightwheel",
            OBJECT_ZIPS,
            assets_root / "objects" / "lightwheel",
        ),
    }
    selected_prefixes = {groups[name][0] for name in selected}
    _require_manifests_for_preserved_archives(
        previous_archives,
        selected_prefixes,
    )
    pending: list[tuple[str, ZipInstallRequest]] = []
    for name in selected:
        prefix, filenames, target_dir = groups[name]
        pending.extend(
            prepare_group(
                prefix,
                filenames,
                target_dir,
                download_fn=download_fn,
                previous_archives=previous_archives,
            )
        )

    results = install_zip_batch_exact(
        tuple(request for _remote_path, request in pending)
    )
    installed: dict[str, dict[str, object]] = {}
    for (remote_path, _request), result in zip(
        pending,
        results,
        strict=True,
    ):
        installed[remote_path] = {
            "sha256": result.sha256,
            "roots": list(result.roots),
            "file_count": result.file_count,
            "files": list(result.files),
        }

    archives = {
        filename: record
        for filename, record in previous_archives.items()
        if not any(
            filename.startswith(f"{prefix}/")
            for prefix in selected_prefixes
        )
    }
    archives.update(installed)
    marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "repository": HF_REPO,
        "revision": HF_REVISION,
        "asset_groups": [
            name
            for name, (prefix, _filenames, _target) in groups.items()
            if any(
                filename.startswith(f"{prefix}/")
                for filename in archives
            )
        ],
        "archives": archives,
    }
    write_json_atomic(marker_path, marker)


def prepare_group(
    prefix: str,
    filenames: tuple[str, ...],
    target_dir: Path,
    *,
    download_fn: Callable[..., str],
    previous_archives: Mapping[str, object],
) -> list[tuple[str, ZipInstallRequest]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[str, ZipInstallRequest]] = []
    for filename in filenames:
        remote_path = f"{prefix}/{filename}"
        print(f"Downloading {remote_path} from {HF_REPO}@{HF_REVISION}")
        archive = Path(
            download_fn(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=remote_path,
                revision=HF_REVISION,
            )
        )
        print(f"Staging {filename} for {target_dir}")
        pending.append(
            (
                remote_path,
                ZipInstallRequest(
                    archive=archive,
                    target=target_dir,
                    previous_files=_previous_files(
                        previous_archives,
                        remote_path,
                    ),
                ),
            )
        )
    return pending


def _load_previous_archives(
    marker_path: Path,
) -> Mapping[str, object]:
    if not marker_path.is_file():
        return {}
    # Asset roots persist across installer versions; following a marker symlink can read ownership from another tree.
    if marker_path.is_symlink():
        raise ValueError(
            f"Lightwheel asset marker is a symbolic link: {marker_path}"
        )
    # Existing roots may carry a legacy marker, but its ownership table must remain a mapping.
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    archives = marker.get("archives") if isinstance(marker, Mapping) else None
    # Transactional replacement requires a filename-to-ownership mapping.
    if not isinstance(archives, Mapping):
        raise ValueError(
            f"Invalid Lightwheel asset marker: {marker_path}"
        )
    schema_version = marker.get("schema_version")
    # Unknown marker semantics could merge files owned by an incompatible installer.
    if schema_version not in {None, MARKER_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported Lightwheel marker schema: {schema_version}"
        )
    for field, expected in (
        ("repository", HF_REPO),
        ("revision", HF_REVISION),
    ):
        actual = marker.get(field)
        # Reusing ownership data from another source silently mixes asset revisions.
        if actual is not None and actual != expected:
            raise ValueError(
                f"Lightwheel asset marker {field} mismatch: {actual}"
            )
    if schema_version == MARKER_SCHEMA_VERSION:
        for filename, record in archives.items():
            # Current marker records are the ownership contract consumed by transactional replacement.
            if not isinstance(filename, str) or not isinstance(
                record, Mapping
            ):
                raise ValueError(
                    f"Invalid Lightwheel archive record: {filename}"
                )
            # Missing ownership leaves stale archive files in place after replacement.
            if "files" not in record:
                raise ValueError(
                    "Lightwheel archive record lacks a file manifest: "
                    f"{filename}"
                )
            _previous_files(archives, filename)
    return archives


def _require_manifests_for_preserved_archives(
    archives: Mapping[str, object],
    selected_prefixes: set[str],
) -> None:
    for filename, record in archives.items():
        selected = any(
            filename.startswith(f"{prefix}/")
            for prefix in selected_prefixes
        )
        if selected:
            continue
        # Preserved records come from legacy markers and require explicit file ownership.
        if not isinstance(record, Mapping) or "files" not in record:
            raise ValueError(
                "Legacy Lightwheel marker cannot preserve an unselected "
                f"archive without a file manifest: {filename}; install all"
            )


def _previous_files(
    archives: Mapping[str, object],
    filename: str,
) -> tuple[str, ...]:
    record = archives.get(filename)
    # Legacy markers can omit an archive record; current mappings carry an optional file list.
    if not isinstance(record, Mapping):
        return ()
    if "files" not in record:
        return ()
    files = record["files"]
    # Transactional cleanup consumes these values as relative paths, not arbitrary JSON.
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
