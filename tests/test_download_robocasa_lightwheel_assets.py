import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import download_robocasa_lightwheel_assets


def _zip(path: Path, member: str, content: bytes = b"asset") -> Path:
    with ZipFile(path, "w") as archive:
        archive.writestr(member, content)
    return path


def test_downloads_pinned_lightwheel_groups_and_writes_marker(
    monkeypatch,
    tmp_path,
):
    archives = {
        "fixtures_lightwheel/cabinets.zip": _zip(
            tmp_path / "cabinets.zip",
            "cabinet/model.xml",
        ),
        "objects_lightwheel/peeler.zip": _zip(
            tmp_path / "peeler.zip",
            "peeler/model.xml",
        ),
    }
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(archives[kwargs["filename"]])

    monkeypatch.setattr(
        download_robocasa_lightwheel_assets,
        "FIXTURE_ZIPS",
        ("cabinets.zip",),
    )
    monkeypatch.setattr(
        download_robocasa_lightwheel_assets,
        "OBJECT_ZIPS",
        ("peeler.zip",),
    )
    assets_root = tmp_path / "assets"
    stale_fixture = assets_root / "fixtures/cabinet/stale.txt"
    stale_fixture.parent.mkdir(parents=True)
    stale_fixture.write_text("stale", encoding="utf-8")
    (
        assets_root
        / download_robocasa_lightwheel_assets.MARKER_NAME
    ).write_text(
        json.dumps(
            {
                "archives": {
                    "fixtures_lightwheel/cabinets.zip": {
                        "files": ["cabinet/stale.txt"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    download_robocasa_lightwheel_assets.download_assets(
        ("fixtures", "objects"),
        assets_root=assets_root,
        download_fn=fake_download,
    )

    assert (assets_root / "fixtures/cabinet/model.xml").read_bytes() == b"asset"
    assert (
        assets_root / "objects/lightwheel/peeler/model.xml"
    ).read_bytes() == b"asset"
    assert not stale_fixture.exists()
    assert all(
        call["repo_id"] == download_robocasa_lightwheel_assets.HF_REPO
        and call["revision"] == download_robocasa_lightwheel_assets.HF_REVISION
        and call["repo_type"] == "dataset"
        for call in calls
    )
    marker = json.loads(
        (
            assets_root
            / download_robocasa_lightwheel_assets.MARKER_NAME
        ).read_text(encoding="utf-8")
    )
    assert marker == {
        "schema_version": 1,
        "repository": (
            "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
        ),
        "revision": "420a04af939c34873e6839a586b70844baf28aab",
        "asset_groups": ["fixtures", "objects"],
        "archives": {
            "fixtures_lightwheel/cabinets.zip": {
                "sha256": hashlib.sha256(
                    archives[
                        "fixtures_lightwheel/cabinets.zip"
                    ].read_bytes()
                ).hexdigest(),
                "roots": ["cabinet"],
                "file_count": 1,
                "files": ["cabinet/model.xml"],
            },
            "objects_lightwheel/peeler.zip": {
                "sha256": hashlib.sha256(
                    archives[
                        "objects_lightwheel/peeler.zip"
                    ].read_bytes()
                ).hexdigest(),
                "roots": ["peeler"],
                "file_count": 1,
                "files": ["peeler/model.xml"],
            },
        },
    }


def test_rejects_lightwheel_archive_path_traversal(monkeypatch, tmp_path):
    archive = _zip(tmp_path / "cabinets.zip", "../escape.txt")
    monkeypatch.setattr(
        download_robocasa_lightwheel_assets,
        "FIXTURE_ZIPS",
        ("cabinets.zip",),
    )

    with pytest.raises(ValueError, match="escapes extraction directory"):
        download_robocasa_lightwheel_assets.download_assets(
            ("fixtures",),
            assets_root=tmp_path / "assets",
            download_fn=lambda **_kwargs: str(archive),
        )

    assert not (tmp_path / "escape.txt").exists()


def test_previous_lightwheel_marker_without_file_manifest_is_migrated():
    assert download_robocasa_lightwheel_assets._previous_files(
        {
            "fixtures_lightwheel/cabinets.zip": {
                "sha256": "legacy",
            }
        },
        "fixtures_lightwheel/cabinets.zip",
    ) == ()


def test_partial_lightwheel_install_preserves_other_group_ownership(
    monkeypatch,
    tmp_path,
):
    fixture_archive = _zip(
        tmp_path / "cabinets.zip",
        "cabinet/new.xml",
    )
    monkeypatch.setattr(
        download_robocasa_lightwheel_assets,
        "FIXTURE_ZIPS",
        ("cabinets.zip",),
    )
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    marker_path = (
        assets_root
        / download_robocasa_lightwheel_assets.MARKER_NAME
    )
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": (
                    download_robocasa_lightwheel_assets.HF_REPO
                ),
                "revision": (
                    download_robocasa_lightwheel_assets.HF_REVISION
                ),
                "asset_groups": ["fixtures", "objects"],
                "archives": {
                    "fixtures_lightwheel/cabinets.zip": {
                        "files": ["cabinet/old.xml"],
                    },
                    "objects_lightwheel/peeler.zip": {
                        "files": ["peeler/keep.xml"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    download_robocasa_lightwheel_assets.download_assets(
        ("fixtures",),
        assets_root=assets_root,
        download_fn=lambda **_kwargs: str(fixture_archive),
    )

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["asset_groups"] == ["fixtures", "objects"]
    assert set(marker["archives"]) == {
        "fixtures_lightwheel/cabinets.zip",
        "objects_lightwheel/peeler.zip",
    }
    assert marker["archives"]["objects_lightwheel/peeler.zip"] == {
        "files": ["peeler/keep.xml"]
    }
