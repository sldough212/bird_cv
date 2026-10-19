import dataclasses
import json
import re
import sys
from pathlib import Path

import cv2
import msgspec
import numpy as np
import polars as pl
import pytest

from bird_cv.core.labels import NO_BIRD
from bird_cv.core.layout import EvalRun, Layout
from bird_cv.core.manifest import new_manifest
from bird_cv.core.schemas import PREDICTION_SCHEMA
from bird_cv.evaluation.e2e import sweep_settings
from bird_cv.data.check_fps import frame_rate_table
from bird_cv.pipelines.build_datasets import build_datasets
from bird_cv.pipelines.config import API_KEY_ENV_VAR, load_config
from bird_cv.pipelines.evaluate import run_evaluate
from bird_cv.pipelines.infer import split_run
from bird_cv.pipelines.prepare import run_prepare
from bird_cv.preprocessing.standardize import expected_standard_frames

sys.path.insert(0, str(Path(__file__).parent))
from test_data import CONFIG, classify_task, detect_task  # noqa: E402


def test_expected_standard_frames_matches_ffmpeg() -> None:
    # (header frames, header fps) -> frames ffmpeg's fps=15 filter wrote
    for frames, fps, written in [
        (101, 25, 61),
        (1502, 25, 901),
        (97, 30, 49),
        (95, 30000 / 1001, 48),
        (37, 12, 46),
        (50, 24, 31),
        (40, 15, 40),
    ]:
        assert expected_standard_frames(frames, fps) == written
    assert expected_standard_frames(1502, 25, max_frames=900) == 900

    # Label Studio at 25 fps: reported misaligned (and excluded from ground truth)
    task = detect_task("a", "v", "0", [(1, 2)], n=1502)
    task["annotations"][0]["result"][0]["value"]["duration"] = 60.07
    assert frame_rate_table(Path("/nonexistent"), [task])["aligned"].to_list() == [
        False
    ]


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (16, 16))
    for i in range(n_frames):
        writer.write(np.full((16, 16, 3), i * 10, dtype=np.uint8))
    writer.release()


def test_prepare_build_evaluate_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "test")
    base = tmp_path / "out"
    config = tmp_path / "config.toml"
    raw = tmp_path / "raw"
    text = re.sub(
        r'^path_to_base_output = ".*"$',
        f'path_to_base_output = "{base}"',
        CONFIG.read_text(),
        flags=re.M,
    )
    text = re.sub(
        r'^path_to_raw_videos = ".*"$',
        f'path_to_raw_videos = "{raw}"',
        text,
        flags=re.M,
    )
    config.write_text(text)
    cfg = load_config(config)
    cameras, n = "abcd", 20
    for c in cameras:  # "v" is labeled, "u" isn't; headers unreadable (no counts)
        for video in ("v", "u"):
            (raw / c).mkdir(parents=True, exist_ok=True)
            (raw / c / f"{video}.mp4").touch()
    # Boxes from LS frame 6: crop frame k is canonical k + 5, not k
    detect = [detect_task(c, "v", "0", [(6, n)], n=n) for c in cameras]
    classify = [
        classify_task(c, "v", "0", [("preening", 1, 7), ("resting", 8, 15)])
        for c in cameras
    ]
    # Media outside path_to_base_output, found from the exports' URLs
    for tasks, kind in ((detect, "cage_videos"), (classify, "bird_crops")):
        for c, task in zip(cameras, tasks):
            media = tmp_path / "legacy" / kind / c / "v" / "0.mp4"
            _write_video(media, n)
            task["data"]["video"] = f"/data/local-files/?d={str(media)[1:]}"
    (tmp_path / "detect.json").write_text(json.dumps(detect))
    (tmp_path / "classify.json").write_text(json.dumps(classify))
    run_prepare(
        cfg,
        detect_export=tmp_path / "detect.json",
        classify_export=tmp_path / "classify.json",
    )
    layout = Layout(base)
    prepared = layout.read_current()
    truth = pl.read_parquet(
        layout.ground_truth(prepared.snapshot_hash, prepared.gt_version)
    )
    boxed = set(
        truth.filter(pl.col("bbox_left").is_not_null())
        .select("camera_id", "frame")
        .iter_rows()
    )

    # Missing media fails before any dataset is written
    moved = tmp_path / "legacy" / "cage_videos" / "a" / "v" / "0.mp4"
    moved.rename(moved.with_suffix(".bak"))
    with pytest.raises(FileNotFoundError, match="1 selected media"):
        build_datasets(cfg, prepared.id, folds=None, processes=1)
    assert not list(base.rglob("datasets"))
    moved.with_suffix(".bak").rename(moved)

    built = build_datasets(cfg, prepared.id, folds=None, processes=1)
    assert build_datasets(cfg, prepared.id, folds=None, processes=1) == built

    for fold in (None, 0):
        for role, spec in cfg.models.roles().items():
            out = layout.dataset(prepared, role, spec.kind, fold)
            split = dict(
                pl.read_parquet(
                    layout.dataset_split(prepared, role, spec.kind, fold)
                ).iter_rows()
            )
            # detector: images/{split}/{camera}_v_0_{frame}.jpg;
            # classifier: {split}/{label}/{camera}_v_0_s{i}_clip000/
            files = (
                (out / "images").glob("*/*.jpg")
                if role == "detector"
                else out.glob("*/*/*_clip*")
            )
            seen: dict[str, set[str]] = {}
            for f in files:
                split_dir = f.parent.name if role == "detector" else f.parents[1].name
                seen.setdefault(f.name.split("_")[0], set()).add(split_dir)
            assert seen, (role, fold)
            assert all(dirs == {split[c]} for c, dirs in seen.items()), (seen, split)

            if role == "detector":
                # Image stem frame F is canonical F - 1; labeled iff boxed
                selected = set(
                    pl.read_parquet(
                        layout.dataset_selection(prepared, role, spec.kind, fold)
                    )
                    .select("camera_id", "frame")
                    .iter_rows()
                )
                for image in (out / "images").glob("*/*.jpg"):
                    camera, frame = image.stem.split("_")[0], int(image.stem[-5:]) - 1
                    assert (camera, frame) in selected
                    label = out / "labels" / image.parent.name / f"{image.stem}.txt"
                    assert bool(label.read_text().strip()) == ((camera, frame) in boxed)

    # Evaluate through the layout: setting 1 perfect, the rest empty
    run_id = "r1"
    predictions = truth.with_columns(
        label=pl.col("label").fill_null(NO_BIRD), confidence=1.0, det_conf=1.0
    ).select(list(PREDICTION_SCHEMA))
    settings = sweep_settings(
        cfg.evaluate.sweep_gap_fill_frames, cfg.evaluate.sweep_det_conf
    )
    manifest = new_manifest("infer", config={"prepared": dataclasses.asdict(prepared)})
    runs = [
        (layout.sweep_infer_run(run_id, g, c), i == 1)
        for i, (g, c) in enumerate(settings)
    ] + [(layout.test_infer_run(run_id), True)]
    for run, perfect in runs:
        preds = (
            predictions if perfect else predictions.with_columns(label=pl.lit(NO_BIRD))
        )
        run.predictions("all", "all").parent.mkdir(parents=True)
        preds.write_parquet(run.predictions("all", "all"))
        run.task_manifest(0).write_bytes(msgspec.json.encode(manifest))
    sweep = EvalRun(run_evaluate(cfg, "sweep", run_id, prepared_id=prepared.id))
    best = json.loads(sweep.best_setting.read_text())
    assert (best["gap_fill_frames"], best["det_conf"]) == settings[1]
    test = EvalRun(run_evaluate(cfg, "test", run_id, prepared_id=prepared.id))
    assert json.loads(test.metrics.read_text())["macro_f1"] == 1.0

    # infer.py --split: labeled videos of the split's cameras, setting dirs
    frozen = dict(
        pl.read_parquet(layout.split(prepared.split_version))
        .select("camera_id", "split")
        .iter_rows()
    )
    for split_name, index, setting, run in (
        ("val", 2, settings[2], layout.sweep_infer_run(run_id, *settings[2])),
        ("test", None, settings[1], layout.test_infer_run(run_id)),
    ):
        videos, run_dir, infer_cfg, _ = split_run(
            cfg, run_id, split_name, index, prepared.id
        )
        assert videos
        assert sorted(v.relative_to(raw).as_posix() for v in videos) == sorted(
            f"{c}/v.mp4" for c in cameras if frozen[c] == split_name
        )
        assert run_dir == run.dir
        assert (infer_cfg.gap_fill_frames, infer_cfg.det_conf) == setting
