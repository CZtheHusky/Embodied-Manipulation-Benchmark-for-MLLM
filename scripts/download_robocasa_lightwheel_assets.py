#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable
from zipfile import ZipFile


HF_REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
HF_REVISION = "420a04af939c34873e6839a586b70844baf28aab"
MARKER_NAME = ".robocasa-lightwheel-revision.json"

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
    unknown = sorted(set(selected) - {"fixtures", "objects"})
    if unknown:
        raise ValueError(f"Unknown Lightwheel asset groups: {unknown}")
    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download

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
    downloaded_archives: list[str] = []
    for name in selected:
        prefix, filenames, target_dir = groups[name]
        downloaded_archives.extend(
            download_group(
                prefix,
                filenames,
                target_dir,
                download_fn=download_fn,
            )
        )

    marker = {
        "repository": HF_REPO,
        "revision": HF_REVISION,
        "asset_groups": list(selected),
        "archives": downloaded_archives,
    }
    (assets_root / MARKER_NAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def download_group(
    prefix: str,
    filenames: tuple[str, ...],
    target_dir: Path,
    *,
    download_fn: Callable[..., str],
) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
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
        print(f"Extracting {filename} into {target_dir}")
        _extract_zip(archive, target_dir)
        downloaded.append(remote_path)
    return downloaded


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
