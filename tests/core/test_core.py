from pathlib import Path

import pytest

from bird_cv.core.frames import canonical_to_ls, ls_to_canonical
from bird_cv.core.ids import parse_media_path, parse_raw_video_path
from bird_cv.core.labels import Behaviors
from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.pipelines.config import API_KEY_ENV_VAR, load_config

CONFIG = Path(__file__).parents[2] / "bird_cv/pipelines/configs/config.toml"


def test_label_studio_frames_round_trip() -> None:
    assert ls_to_canonical(1) == 0
    assert canonical_to_ls(ls_to_canonical(46)) == 46


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


def test_ids_match_between_label_studio_and_raw_paths() -> None:
    url = "/data/local-files/?d=gscratch/x/cage_videos/I35%2CL15/IMG_0016/0.mp4"
    assert parse_media_path(url) == ("I35,L15", "IMG_0016", "0")
    assert parse_raw_video_path("videos/I35,L15/IMG_0016.MOV") == (
        "I35,L15",
        "IMG_0016",
    )


def test_optional_config_table_rejects_typos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "k")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(CONFIG.read_text() + "\n[infer]\ngap_fil_frames = 3\n")
    with pytest.raises(ValueError, match="gap_fil_frames"):
        load_config(cfg_path)


def test_config_api_key_optional_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    assert load_config(CONFIG, require_api_key=False).api_key == ""


def test_behaviors_validate_vocabulary() -> None:
    vocab = Behaviors(names=("a", "resting"))
    assert vocab.classes == ("a", "resting", "no_bird")
    assert vocab.active == ("a",)
    with pytest.raises(ValueError, match="sorted"):
        Behaviors(names=("resting", "a"))
    with pytest.raises(ValueError, match="resting"):
        Behaviors(names=("a", "b"))
    with pytest.raises(ValueError, match="Unknown class"):
        vocab.validate("b")
