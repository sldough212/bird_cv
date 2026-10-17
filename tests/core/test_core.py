from pathlib import Path

import pytest

from bird_cv.core.frames import FrameMap, canonical_to_ls, ls_to_canonical
from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.pipelines.config import API_KEY_ENV_VAR, load_config

CONFIG = Path(__file__).parents[2] / "bird_cv/pipelines/configs/config.toml"


def test_label_studio_frames_round_trip() -> None:
    assert ls_to_canonical(1) == 0
    assert canonical_to_ls(ls_to_canonical(46)) == 46


def test_frame_map_round_trips_and_rejects_unordered(tmp_path: Path) -> None:
    fmap = FrameMap((45, 46, 219))
    fmap.write(tmp_path / "0.frames.json")
    assert FrameMap.read(tmp_path / "0.frames.json").to_canonical(2) == 219
    with pytest.raises(ValueError):
        FrameMap((3, 2))


def test_manifest_round_trips(tmp_path: Path) -> None:
    manifest = new_manifest("prepare", snapshot_hash="abc", split_version=1)
    write_manifest(manifest, tmp_path)
    assert read_manifest(tmp_path) == manifest


def test_config_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "from-env")
    cfg = load_config(CONFIG)
    assert cfg.api_key == "from-env"
    assert cfg.infer.gap_fill_frames == 8


def test_config_without_api_key_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=API_KEY_ENV_VAR):
        load_config(CONFIG)
