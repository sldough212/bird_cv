"""Train YOLO or VideoMAE from generic pretrained weights (ADR 0004).

On the frozen split, YOLO is Ray-tuned and selected on val presence F1 and
VideoMAE is Ray-tuned (ASHA) and selected on val window macro-F1 (ADR
0005). With ``--fold k`` a model is retrained on CV fold ``k``'s dataset
with fixed hyperparameters from a frozen-split run (ADR 0001), no tuning.
``--smoke`` shrinks tuning and epochs to 2 for a first cluster run (a smoke
test; never report its scores). The ``paths`` mode prints the resolved
output paths for a run id as shell assignments, without creating anything.

Usage::

    python bird_cv/pipelines/train.py CONFIG {yolo,videomae} [--run-id ID] [--smoke]
    python bird_cv/pipelines/train.py CONFIG {yolo,videomae} --fold K \\
        --hyperparameters RUN_DIR/best_hyperparameters.json [--run-id ID]
    python bird_cv/pipelines/train.py CONFIG paths --run-id ID
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.evaluation.cv import check_fold_cameras
from bird_cv.pipelines.config import BirdCVConfig, load_config, resolve_run_dir

MODELS = ("yolo", "videomae")

FOLD_SPLIT_FILE = "split.parquet"
"""Written into each fold dataset by whatever builds it: the
:func:`bird_cv.evaluation.cv.fold_split` output it was built from."""


def smoke_config(cfg: BirdCVConfig) -> BirdCVConfig:
    """Copy of ``cfg`` with 2 tuning trials and 2 epochs for both models."""
    train = dataclasses.replace(
        cfg.train,
        yolo_tune_iterations=2,
        yolo_epochs=2,
        mae_tune_trials=2,
        mae_epochs=2,
        mae_tune_grace_period=1,
    )
    return dataclasses.replace(cfg, train=train)


def _dataset_dir(cfg: BirdCVConfig, relative: Path, fold: int | None) -> Path:
    root = cfg.path_to_base_output / relative
    return root / f"fold{fold}" if fold is not None else root


def _run_base(cfg: BirdCVConfig, model: str, fold: int | None) -> Path:
    base = cfg.path_to_base_output / cfg.train.output / model
    return base / f"fold{fold}" if fold is not None else base


def _run_dir(
    cfg: BirdCVConfig, model: str, fold: int | None, run_id: str | None
) -> Path:
    return resolve_run_dir(_run_base(cfg, model, fold), run_id)


def _write_manifest(
    kind: str,
    cfg: BirdCVConfig,
    run_dir: Path,
    dataset: Path,
    fold: int | None,
    extra_inputs: dict[str, str] | None = None,
) -> None:
    """Manifest carrying the dataset's snapshot hash and split version, if known."""
    upstream = read_manifest(dataset) if (dataset / "manifest.json").exists() else None
    write_manifest(
        new_manifest(
            kind=kind,
            snapshot_hash=upstream.snapshot_hash if upstream else None,
            split_version=upstream.split_version if upstream else None,
            config={"train": dataclasses.asdict(cfg.train), "fold": fold},
            inputs={"dataset": str(dataset)} | (extra_inputs or {}),
        ),
        run_dir,
    )


def _train_val_cameras(dataset: Path, yolo_layout: bool) -> set[str]:
    """Cameras in a fold dataset's train and val data.

    Read from ``{dataset}/split.parquet`` when present and, for the YOLO
    layout (``images/{split}/{camera_id}/...``), from the image directories.
    """
    cameras: set[str] = set()
    split_file = dataset / FOLD_SPLIT_FILE
    if split_file.exists():
        split = pl.read_parquet(split_file)
        cameras |= set(
            split.filter(pl.col("split").is_in(["train", "val"]))["camera_id"]
        )
    if yolo_layout:
        for name in ("train", "val"):
            split_dir = dataset / "images" / name
            if split_dir.is_dir():
                cameras |= {d.name for d in split_dir.iterdir() if d.is_dir()}
    return cameras


def _check_fold(
    cfg: BirdCVConfig,
    fold: int | None,
    hyperparameters: dict | None,
    dataset: Path,
    yolo_layout: bool,
) -> None:
    """Folds reuse frozen-split hyperparameters and never train on test cameras."""
    if fold is None:
        return
    if hyperparameters is None:
        raise ValueError(
            "CV folds reuse the frozen-split hyperparameters; pass --hyperparameters"
        )
    cameras = _train_val_cameras(dataset, yolo_layout)
    if not cameras:
        raise ValueError(
            f"Can't tell which cameras {dataset} trains on: write its "
            f"fold_split output to {dataset / FOLD_SPLIT_FILE}"
        )
    cv_folds = pl.read_parquet(cfg.path_to_base_output / cfg.evaluate.cv_folds)
    check_fold_cameras(cv_folds, fold, cameras)


def run_train_yolo(
    cfg: BirdCVConfig,
    run_id: str | None = None,
    fold: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> Path:
    """Train YOLO, select on val presence F1, write ``best.pt`` and a manifest.

    Args:
        cfg: Pipeline config; uses ``[train]``, ``infer.det_conf`` (scoring
            threshold) and ``evaluate.iou_threshold``.
        run_id: Fixed run directory name; timestamped if ``None``.
        fold: CV fold to train on; ``None`` for the frozen split.
        hyperparameters: Fixed training arguments (a frozen-split run's
            ``best_hyperparameters.json``). Disables tuning.

    Returns:
        The run directory.
    """
    from bird_cv.detection.train_yolo import train_select_yolo

    t = cfg.train
    dataset = _dataset_dir(cfg, t.yolo_dataset, fold)
    _check_fold(cfg, fold, hyperparameters, dataset, yolo_layout=True)
    run_dir = _run_dir(cfg, "yolo", fold, run_id)
    best = train_select_yolo(
        dataset_root=dataset,
        run_dir=run_dir,
        model_path=t.yolo_pretrained,
        epochs=t.yolo_epochs,
        device=t.device,
        tune=t.yolo_tune and hyperparameters is None,
        iterations=t.yolo_tune_iterations,
        hyperparameters=hyperparameters,
        select_conf=cfg.infer.det_conf,
        iou_threshold=cfg.evaluate.iou_threshold,
    )
    _write_manifest(
        "train_yolo",
        cfg,
        run_dir,
        dataset,
        fold,
        {"pretrained": t.yolo_pretrained, "selected": best["checkpoint"]},
    )
    return run_dir


def run_train_videomae(
    cfg: BirdCVConfig,
    run_id: str | None = None,
    fold: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> Path:
    """Tune (optionally) and train VideoMAE, selecting epochs on val window macro-F1.

    The checkpoint is written to ``{run_dir}/videomae/best``.

    Args:
        cfg: Pipeline config; uses ``[train]``, and ``evaluate.cv_folds``
            for folds.
        run_id: Fixed run directory name; timestamped if ``None``.
        fold: CV fold to train on; ``None`` for the frozen split.
        hyperparameters: Fixed hyperparameters (a frozen-split run's
            ``best_hyperparameters.json``). Disables tuning.

    Returns:
        The run directory.
    """
    from bird_cv.classification.train_video_model import train_video_model
    from bird_cv.classification.tune_video_model import TUNED_KEYS, tune_video_model

    t = cfg.train
    clips = _dataset_dir(cfg, t.mae_clips, fold)
    _check_fold(cfg, fold, hyperparameters, clips, yolo_layout=False)
    run_dir = _run_dir(cfg, "videomae", fold, run_id)

    if hyperparameters is None and t.mae_tune:
        hyperparameters = tune_video_model(
            clips_root=clips,
            output_root=run_dir / "tune",
            model_checkpoint=t.mae_pretrained,
            num_samples=t.mae_tune_trials,
            epochs=t.mae_epochs,
            grace_period=t.mae_tune_grace_period,
            num_frames=t.mae_num_frames,
            batch_size=t.mae_batch_size,
            cpus_per_trial=t.mae_cpus_per_trial,
        )
    # Untuned runs pass nothing, so train_video_model's own defaults apply;
    # the empty JSON makes CV folds reuse those same defaults.
    given = hyperparameters or {}
    params: dict[str, Any] = {k: given[k] for k in TUNED_KEYS if k in given}
    (run_dir / "best_hyperparameters.json").write_text(json.dumps(params, indent=2))

    result = train_video_model(
        clips_root=clips,
        output_root=run_dir,
        model_checkpoint=t.mae_pretrained,  # type: ignore[arg-type]
        output_name="videomae",
        num_frames=t.mae_num_frames,
        epochs=t.mae_epochs,
        batch_size=t.mae_batch_size,
        selection_metric="macro_f1",
        **params,
    )
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    _write_manifest(
        "train_videomae", cfg, run_dir, clips, fold, {"pretrained": t.mae_pretrained}
    )
    return run_dir


def print_paths(cfg: BirdCVConfig, run_id: str) -> None:
    """Print a frozen-split run's resolved paths as shell assignments."""
    paths = {
        "BASE": cfg.path_to_base_output,
        "YOLO_RUN_DIR": _run_base(cfg, "yolo", None) / run_id,
        "YOLO_BEST": _run_base(cfg, "yolo", None) / run_id / "best.pt",
        "MAE_RUN_DIR": _run_base(cfg, "videomae", None) / run_id,
        "MAE_BEST": _run_base(cfg, "videomae", None) / run_id / "videomae" / "best",
    }
    for name, path in paths.items():
        print(f"{name}='{path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument("model", choices=MODELS + ("paths",))
    parser.add_argument("--run-id", default=None, help="Run directory name")
    parser.add_argument("--fold", type=int, default=None, help="CV fold to train on")
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="best_hyperparameters.json from a frozen-split run (disables tuning)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="2 tuning trials and 2 epochs"
    )
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.model == "paths":
        if not args.run_id:
            parser.error("paths needs --run-id")
        print_paths(cfg, args.run_id)
        return

    hyperparameters = (
        json.loads(args.hyperparameters.read_text()) if args.hyperparameters else None
    )
    run = run_train_yolo if args.model == "yolo" else run_train_videomae
    run_dir = run(cfg, args.run_id, args.fold, hyperparameters)
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
