import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

from bird_cv.core.layout import Layout
from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.pipelines.evaluate import check_dataset
from bird_cv.inference.registry import inference_models
from bird_cv.pipelines.config import ModelsConfig, load_config
from bird_cv.training.registry import (
    VideoMAETrainer,
    YoloTrainer,
    build_search_space,
    get_trainer,
)

CONFIG = Path(__file__).parents[1] / "bird_cv/pipelines/configs/config.toml"


def _config(tmp_path: Path, detector: str = "") -> Path:
    """The committed config under ``tmp_path``, ``detector`` added to
    [models.detector]."""
    text = re.sub(
        r'^path_to_base_output = ".*"\s*$',
        f'path_to_base_output = "{tmp_path}"',
        CONFIG.read_text(),
        flags=re.M,
    )
    text = text.replace("\n[models.detector]\n", f"\n[models.detector]\n{detector}\n")
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def _without_checkpoints(models: ModelsConfig) -> ModelsConfig:
    """``train`` minus the committed checkpoints and inference options."""
    specs: dict[str, Any] = {
        role: dataclasses.replace(spec, checkpoint=None, infer_options={})
        for role, spec in models.roles().items()
    }
    return dataclasses.replace(models, **specs)


def test_train_tables_merge_onto_defaults(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path, "epochs = 5"), require_api_key=False)
    defaults = ModelsConfig()
    assert cfg.models.detector.epochs == 5
    assert cfg.models.detector.kind == defaults.detector.kind
    assert cfg.models.detector.pretrained == defaults.detector.pretrained
    assert cfg.models.detector.tune_trials == defaults.detector.tune_trials
    assert _without_checkpoints(cfg.models).classifier == defaults.classifier

    with pytest.raises(ValueError, match="models.detector"):
        load_config(_config(tmp_path, "epoch = 5"), require_api_key=False)

    models = load_config(_config(tmp_path), require_api_key=False).models
    assert _without_checkpoints(models) == defaults


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
    assert models.checkpoints["detector"] == tmp_path / "models/detector/r1/best.pt"

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


def test_evaluate_rejects_runs_on_other_datasets(tmp_path: Path) -> None:
    recorded = {"id": "d1", "key": {"split_version": 3}}
    path = write_manifest(new_manifest("infer", config={"dataset": recorded}), tmp_path)
    # A reused (linked) dataset id with the same key matches
    check_dataset({"id": "d2", "key": {"split_version": 3}}, [path], "run")
    with pytest.raises(ValueError, match="dataset d1"):
        check_dataset({"id": "d3", "key": {"split_version": 2}}, [path], "run")


def test_cage_detector_role_is_opt_in_and_feeds_yolo_cages(tmp_path: Path) -> None:
    cfg = load_config(_config(tmp_path), require_api_key=False)
    assert "cage_detector" not in cfg.models.roles()
    path = _config(tmp_path)
    swap = '[infer.segmenter]\nkind = "yolo_cages"\noptions = { expected_cages = 4 }\n'
    path.write_text(path.read_text().replace("[infer.segmenter]\n", swap, 1))
    with pytest.raises(ValueError, match=r"needs \[models.cage_detector\]"):
        inference_models(load_config(path, require_api_key=False), train_run_id="r1")

    path.write_text(path.read_text() + "\n[models.cage_detector]\nepochs = 5\n")
    cfg = load_config(path, require_api_key=False)
    spec = cfg.models.roles()["cage_detector"]
    assert (spec.kind, spec.pretrained, spec.epochs) == (
        "yolo_seg",
        "yolo11s-seg.pt",
        5,
    )
    assert cfg.prepare.path_to_cage_masks == cfg.path_to_segmentation_output
    layout = Layout(tmp_path)
    for role in ("detector", "classifier", "cage_detector"):
        kind = cfg.models.roles()[role].kind
        checkpoint = get_trainer(kind).checkpoint_path(layout.train_run(role, "r1"))
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
    models = inference_models(cfg, train_run_id="r1")
    assert models.context.checkpoint == tmp_path / "models/cage_detector/r1/best.pt"
    models.segmenters().check_cameras(["new_camera"])  # no prompt config needed
