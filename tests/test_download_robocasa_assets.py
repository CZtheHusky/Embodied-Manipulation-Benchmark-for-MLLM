import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import download_robocasa_assets


def _zip(path: Path, member: str, content: bytes = b"asset") -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr(member, content)
    return path


def test_downloads_only_pinned_required_asset_groups(tmp_path):
    archives = {
        "textures.zip": _zip(tmp_path / "textures.zip", "textures/test.png"),
        "objaverse.zip": _zip(
            tmp_path / "objaverse.zip",
            "objaverse/knife/model.xml",
        ),
    }
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(archives[kwargs["filename"]])

    assets_root = tmp_path / "assets"
    stale_texture = assets_root / "textures/stale.txt"
    stale_texture.parent.mkdir(parents=True)
    stale_texture.write_text("stale", encoding="utf-8")
    (assets_root / download_robocasa_assets.MARKER_NAME).write_text(
        json.dumps(
            {
                "archives": {
                    "textures.zip": {
                        "files": ["textures/stale.txt"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    download_robocasa_assets.download_assets(
        ("textures", "objaverse"),
        assets_root=assets_root,
        download_fn=fake_download,
    )

    assert (assets_root / "textures/test.png").read_bytes() == b"asset"
    assert (
        assets_root / "objects/objaverse/knife/model.xml"
    ).read_bytes() == b"asset"
    assert not stale_texture.exists()
    assert {call["filename"] for call in calls} == {
        "textures.zip",
        "objaverse.zip",
    }
    assert all(
        call["repo_id"] == download_robocasa_assets.HF_REPO
        and call["revision"] == download_robocasa_assets.HF_REVISION
        and call["repo_type"] == "dataset"
        for call in calls
    )
    marker = json.loads(
        (assets_root / download_robocasa_assets.MARKER_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert marker == {
        "repository": "robocasa/robocasa-assets",
        "revision": "1b92c3d02ca4354984fec961357db0bff7b32166",
        "asset_groups": ["textures", "objaverse"],
        "archives": {
            "textures.zip": {
                "sha256": hashlib.sha256(
                    archives["textures.zip"].read_bytes()
                ).hexdigest(),
                "roots": ["textures"],
                "file_count": 1,
                "files": ["textures/test.png"],
            },
            "objaverse.zip": {
                "sha256": hashlib.sha256(
                    archives["objaverse.zip"].read_bytes()
                ).hexdigest(),
                "roots": ["objaverse"],
                "file_count": 1,
                "files": ["objaverse/knife/model.xml"],
            },
        },
    }


def test_rejects_archive_path_traversal(tmp_path):
    archive = _zip(tmp_path / "textures.zip", "../escape.txt")

    with pytest.raises(ValueError, match="escapes extraction directory"):
        download_robocasa_assets.download_assets(
            ("textures",),
            assets_root=tmp_path / "assets",
            download_fn=lambda **_kwargs: str(archive),
        )

    assert not (tmp_path / "escape.txt").exists()
