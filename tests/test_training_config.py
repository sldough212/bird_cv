import dataclasses
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from bird_cv.core.layout import Layout
from bird_cv.inference.registry import inference_models
from bird_cv.pipelines.config import TrainConfig, load_config
from bird_cv.pipelines.train import check_fold
from bird_cv.training.registry import (
    VideoMAETrainer,
    YoloTrainer,
    build_search_space,
    get_trainer,
)

CONFIG = Path(__file__).parents[1] / "bird_cv/pipelines/configs/config.toml"


def _config(tmp_path: Path, detector: str = "") -> Path:
    """The committed config under ``tmp_path``, ``detector`` added to
    [train.detector]."""
    text = re.sub(
        r'^path_to_base_output = ".*"$',
        f'path_to_base_output = "{tmp_path}"',
        CONFIG.read_text(),
        flags=re.M,
    )
    text = text.replace("\n[train.detector]\n", f"\n[train.detector]\n{detector}\n")
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def _without_checkpoints(train: TrainConfig) -> TrainConfig:
    """``train`` minus the committed checkpoints and inference options."""
    specs: dict[str, Any] = {
        role: dataclasses.replace(spec, checkpoint=None, infer_options={})
        for role, spec in train.roles().items()
    }
    return dataclasses.replace(train, **specs)


def test_train_tables_merge_onto_defaults(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path, "epochs = 5"), require_api_key=False)
    defaults = TrainConfig()
    assert cfg.train.detector.epochs == 5
    assert cfg.train.detector.kind == defaults.detector.kind
    assert cfg.train.detector.pretrained == defaults.detector.pretrained
    assert cfg.train.detector.tune_trials == defaults.detector.tune_trials
    assert _without_checkpoints(cfg.train).classifier == defaults.classifier

    with pytest.raises(ValueError, match="train.detector"):
        load_config(_config(tmp_path, "epoch = 5"), require_api_key=False)

    train = load_config(_config(tmp_path), require_api_key=False).train
    assert _without_checkpoints(train) == defaults


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
    folds_path = tmp_path / "cv_folds.parquet"
    pl.DataFrame({"camera_id": ["A", "C"], "fold": [0, 1]}).write_parquet(folds_path)

    dataset = tmp_path / "fold1"
    (dataset / "images" / "train" / "C").mkdir(parents=True)  # fold 1's test camera
    split_path = dataset / "split.parquet"
    pl.DataFrame({"camera_id": ["A"], "split": ["train"]}).write_parquet(split_path)

    with pytest.raises(ValueError, match="C"):
        check_fold(
            1,
            {},
            dataset,
            YoloTrainer(),
            split_path=split_path,
            cv_folds_path=folds_path,
        )


def test_inference_models_come_from_config(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path), require_api_key=False)
    with pytest.raises(FileNotFoundError, match="detector.*'r1'"):
        inference_models(cfg, train_run_id="r1")
    layout = Layout(tmp_path)
    for role, trainer in (
        ("detector", YoloTrainer()),
        ("classifier", VideoMAETrainer()),
    ):
        checkpoint = trainer.checkpoint_path(layout.train_run(role, "r1"))
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    models = inference_models(cfg, train_run_id="r1")
    assert models.kinds() == {
        "segmenter": "sam2",
        "detector": "yolo",
        "classifier": "videomae",
    }
    assert models.checkpoints["detector"] == tmp_path / "train/detector/r1/best.pt"

    edits = (
        ("[infer.segmenter]\n", '[infer.segmenter]\nkind = "nope"\n', "nope"),
        ("[infer.segmenter]\n", "[infer.segmenter]\noptions = { iou = 1 }\n", "iou"),
        ("{ tracker_path", "{ trackr_path", "trackr_path"),
        ('checkpoint = "/gscratch', 'checkpoint = "gscratch', "absolute"),
    )
    for old, new, match in edits:
        path = _config(tmp_path)
        path.write_text(path.read_text().replace(old, new, 1))
        with pytest.raises(ValueError, match=match):
            load_config(path, require_api_key=False)
