import re
from pathlib import Path

import polars as pl
import pytest

from bird_cv.pipelines.config import TrainConfig, load_config
from bird_cv.pipelines.train import check_fold
from bird_cv.training.registry import (
    VideoMAETrainer,
    YoloTrainer,
    build_search_space,
    get_trainer,
)

CONFIG = Path(__file__).parents[1] / "bird_cv/pipelines/configs/config.toml"


def _config(tmp_path: Path, extra: str = "") -> Path:
    text = re.sub(
        r'^path_to_base_output = ".*"$',
        f'path_to_base_output = "{tmp_path}"',
        CONFIG.read_text(),
        flags=re.M,
    )
    path = tmp_path / "config.toml"
    path.write_text(text + extra)
    return path


def test_train_tables_merge_onto_defaults(tmp_path: Path) -> None:
    cfg = load_config(
        _config(tmp_path, "\n[train.detector]\nepochs = 5\n"), require_api_key=False
    )
    defaults = TrainConfig()
    assert cfg.train.detector.epochs == 5
    assert cfg.train.detector.kind == defaults.detector.kind
    assert cfg.train.detector.pretrained == defaults.detector.pretrained
    assert cfg.train.detector.tune_trials == defaults.detector.tune_trials
    assert cfg.train.classifier == defaults.classifier

    with pytest.raises(ValueError, match="train.detector"):
        load_config(
            _config(tmp_path, "\n[train.detector]\nepoch = 5\n"),
            require_api_key=False,
        )

    assert load_config(_config(tmp_path), require_api_key=False).train == defaults


def test_registry_and_search_space() -> None:
    assert isinstance(get_trainer("yolo"), YoloTrainer)
    assert isinstance(get_trainer("videomae"), VideoMAETrainer)
    with pytest.raises(ValueError, match="nope"):
        get_trainer("nope")

    pytest.importorskip("ray")
    space = build_search_space(
        {
            "lr": {"type": "loguniform", "lower": 1e-5, "upper": 1e-3},
            "dropout": {"type": "uniform", "lower": 0.0, "upper": 0.5},
            "layers": {"type": "randint", "lower": 1, "upper": 4},
            "freeze": {"type": "choice", "values": [True, False]},
        }
    )
    assert 1e-5 <= space["lr"].sample() <= 1e-3
    assert 0.0 <= space["dropout"].sample() <= 0.5
    assert space["layers"].sample() in {1, 2, 3}
    assert space["freeze"].sample() in {True, False}
    with pytest.raises(ValueError, match="normal"):
        build_search_space({"lr": {"type": "normal", "lower": 0, "upper": 1}})


def test_fold_check_sees_test_camera_in_yolo_layout(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path), require_api_key=False)
    folds_path = tmp_path / cfg.evaluate.cv_folds
    folds_path.parent.mkdir(parents=True)
    pl.DataFrame({"camera_id": ["A", "C"], "fold": [0, 1]}).write_parquet(folds_path)

    dataset = tmp_path / "fold1"
    (dataset / "images" / "train" / "C").mkdir(parents=True)  # fold 1's test camera
    pl.DataFrame({"camera_id": ["A"], "split": ["train"]}).write_parquet(
        dataset / "split.parquet"
    )

    with pytest.raises(ValueError, match="C"):
        check_fold(cfg, 1, {}, dataset, YoloTrainer())
