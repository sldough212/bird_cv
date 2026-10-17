"""Train YOLO or VideoMAE from generic pretrained weights (ADR 0004).

On the frozen split, YOLO is Ray-tuned and selected on val presence F1 and
VideoMAE is Ray-tuned (ASHA) and selected on val window macro-F1 (ADR
0005). With ``--fold k`` a model is retrained on CV fold ``k``'s dataset
with fixed hyperparameters from a frozen-split run (ADR 0001), no tuning.

Usage::

    python bird_cv/pipelines/train.py CONFIG {yolo,videomae} [--run-id ID]
    python bird_cv/pipelines/train.py CONFIG {yolo,videomae} --fold K \\
        --hyperparameters RUN_DIR/best_hyperparameters.json [--run-id ID]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.pipelines.config import BirdCVConfig, load_config, resolve_run_dir

MODELS = ("yolo", "videomae")

# Used when VideoMAE is trained without tuning or tuned hyperparameters;
# matches train_video_model's defaults.
DEFAULT_MAE_HYPERPARAMETERS = {
    "lr": 1e-4,
    "freeze_encoder": True,
    "weight_decay": 0.01,
    "class_weighted_loss": False,
}


def _dataset_dir(cfg: BirdCVConfig, relative: Path, fold: int | None) -> Path:
    root = cfg.path_to_base_output / relative
    return root / f"fold{fold}" if fold is not None else root


def _run_dir(cfg: BirdCVConfig, model: str, fold: int | None, run_id: str | None):
    base = cfg.path_to_base_output / cfg.train.output / model
    return resolve_run_dir(base / f"fold{fold}" if fold is not None else base, run_id)


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
            config={"train": asdict(cfg.train), "fold": fold},
            inputs={"dataset": str(dataset)} | (extra_inputs or {}),
        ),
        run_dir,
    )


def _check_fold_args(fold: int | None, hyperparameters: dict | None) -> None:
    if fold is not None and hyperparameters is None:
        raise ValueError(
            "CV folds reuse the frozen-split hyperparameters; pass --hyperparameters"
        )


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

    _check_fold_args(fold, hyperparameters)
    t = cfg.train
    dataset = _dataset_dir(cfg, t.yolo_dataset, fold)
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
        cfg: Pipeline config; uses ``[train]``.
        run_id: Fixed run directory name; timestamped if ``None``.
        fold: CV fold to train on; ``None`` for the frozen split.
        hyperparameters: Fixed hyperparameters (a frozen-split run's
            ``best_hyperparameters.json``). Disables tuning.

    Returns:
        The run directory.
    """
    from bird_cv.classification.train_video_model import train_video_model
    from bird_cv.classification.tune_video_model import TUNED_KEYS, tune_video_model

    _check_fold_args(fold, hyperparameters)
    t = cfg.train
    clips = _dataset_dir(cfg, t.mae_clips, fold)
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
    params: dict[str, Any] = {
        k: (hyperparameters or DEFAULT_MAE_HYPERPARAMETERS)[k] for k in TUNED_KEYS
    }
    (run_dir / "best_hyperparameters.json").write_text(json.dumps(params, indent=2))

    result = train_video_model(
        clips_root=clips,
        output_root=run_dir,
        model_checkpoint=t.mae_pretrained,  # type: ignore[arg-type]
        output_name="videomae",
        num_frames=t.mae_num_frames,
        epochs=t.mae_epochs,
        batch_size=t.mae_batch_size,
        lr=params["lr"],
        freeze_encoder=params["freeze_encoder"],
        selection_metric="macro_f1",
        weight_decay=params["weight_decay"],
        class_weighted_loss=params["class_weighted_loss"],
    )
    (run_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    _write_manifest(
        "train_videomae", cfg, run_dir, clips, fold, {"pretrained": t.mae_pretrained}
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--run-id", default=None, help="Run directory name")
    parser.add_argument("--fold", type=int, default=None, help="CV fold to train on")
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="best_hyperparameters.json from a frozen-split run (disables tuning)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    hyperparameters = (
        json.loads(args.hyperparameters.read_text()) if args.hyperparameters else None
    )
    run = run_train_yolo if args.model == "yolo" else run_train_videomae
    run_dir = run(cfg, args.run_id, args.fold, hyperparameters)
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
