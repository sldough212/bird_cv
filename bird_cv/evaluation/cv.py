"""3-fold camera cross-validation (ADR 0001).

Every camera is tested exactly once, each fold with the same
hyperparameters (selected on the frozen split). Reported numbers pool every
fold's test predictions into one end-to-end evaluation, plus per-camera and
per-fold breakdowns.
"""

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import polars as pl

from bird_cv.evaluation.e2e import E2EResult, evaluate_e2e


def fold_split(
    cv_folds: pl.DataFrame, fold: int, n_val_cameras: int = 1, seed: int = 42
) -> pl.DataFrame:
    """Train/val/test camera assignment for one CV fold.

    The fold's cameras are the test set. ``n_val_cameras`` of the remaining
    cameras, drawn with ``seed``, are held out for checkpoint selection
    (VideoMAE best epoch, YOLO best checkpoint); the rest are train.

    Args:
        cv_folds: ``CV_FOLD_SCHEMA`` rows.
        fold: The 0-indexed fold to test on.
        n_val_cameras: Non-test cameras held out as val. Defaults to 1.
        seed: Seed for picking val cameras.

    Returns:
        ``camera_id`` and ``split`` (``"train"``, ``"val"`` or ``"test"``),
        sorted by camera.
    """
    test = sorted(cv_folds.filter(pl.col("fold") == fold)["camera_id"].to_list())
    if not test:
        raise ValueError(f"Fold {fold} has no cameras")
    rest = sorted(cv_folds.filter(pl.col("fold") != fold)["camera_id"].to_list())
    val = set(random.Random(seed + fold).sample(rest, min(n_val_cameras, len(rest))))
    return pl.DataFrame(
        {
            "camera_id": test + rest,
            "split": ["test"] * len(test)
            + ["val" if c in val else "train" for c in rest],
        }
    ).sort("camera_id")


def check_fold_cameras(
    cv_folds: pl.DataFrame, fold: int, train_val_cameras: set[str]
) -> None:
    """Raise if a fold's training data includes any of its test cameras.

    Args:
        cv_folds: ``CV_FOLD_SCHEMA`` rows.
        fold: The fold being trained.
        train_val_cameras: Cameras found in the fold's train and val data.

    Raises:
        ValueError: If any of ``train_val_cameras`` is a test camera of
            ``fold``.
    """
    test = set(cv_folds.filter(pl.col("fold") == fold)["camera_id"].to_list())
    if not test:
        raise ValueError(f"Fold {fold} has no cameras")
    leaked = sorted(train_val_cameras & test)
    if leaked:
        raise ValueError(
            f"Fold {fold} trains on its own test cameras {leaked}; build fold "
            "datasets with bird_cv.evaluation.cv.fold_split"
        )


@dataclass(frozen=True)
class CVResult:
    """Cross-validation scores.

    Attributes:
        pooled: All folds' test predictions scored together.
        per_fold: One row per fold: ``fold`` and the summary fields.
    """

    pooled: E2EResult
    per_fold: pl.DataFrame


def evaluate_cv(
    truth: pl.DataFrame,
    cv_folds: pl.DataFrame,
    fold_predictions: Mapping[int, pl.DataFrame],
    iou_threshold: float = 0.5,
) -> CVResult:
    """Score each fold's test predictions, then all of them pooled.

    Args:
        truth: ``GROUND_TRUTH_SCHEMA`` rows for every CV camera.
        cv_folds: ``CV_FOLD_SCHEMA`` rows.
        fold_predictions: ``PREDICTION_SCHEMA`` rows per fold, from that
            fold's models. Rows for cameras outside the fold's test set are
            dropped, so a model is never scored on cameras it trained on.
        iou_threshold: Minimum IoU for a presence hit.

    Returns:
        Pooled and per-fold scores; per-camera scores are in
        ``pooled.per_camera``.
    """
    missing = set(cv_folds["fold"].unique().to_list()) - set(fold_predictions)
    if missing:
        raise ValueError(f"No predictions for folds {sorted(missing)}")

    rows, test_preds = [], []
    for fold, pred in sorted(fold_predictions.items()):
        cameras = cv_folds.filter(pl.col("fold") == fold)["camera_id"]
        fold_truth = truth.filter(pl.col("camera_id").is_in(cameras.implode()))
        fold_pred = pred.filter(pl.col("camera_id").is_in(cameras.implode()))
        result = evaluate_e2e(fold_truth, fold_pred, iou_threshold, with_mot=False)
        rows.append({"fold": fold} | result.summary())
        test_preds.append(fold_pred)

    cv_truth = truth.filter(pl.col("camera_id").is_in(cv_folds["camera_id"].implode()))
    pooled = evaluate_e2e(cv_truth, pl.concat(test_preds), iou_threshold)
    return CVResult(pooled=pooled, per_fold=pl.DataFrame(rows, nan_to_null=True))


def run_cv(
    truth: pl.DataFrame,
    cv_folds: pl.DataFrame,
    train_and_predict: Callable[[int, pl.DataFrame], pl.DataFrame],
    iou_threshold: float = 0.5,
    n_val_cameras: int = 1,
) -> CVResult:
    """Run every fold in-process: train with fixed hyperparameters, predict, score.

    On the cluster the folds run as separate jobs (``pipelines/train.py
    --fold``, then inference) and are scored with :func:`evaluate_cv`; this
    runner is for driving the same loop from one process.

    Args:
        truth: ``GROUND_TRUTH_SCHEMA`` rows for every CV camera.
        cv_folds: ``CV_FOLD_SCHEMA`` rows.
        train_and_predict: Called with ``(fold, split)`` where ``split`` is
            :func:`fold_split`'s output; trains on its train cameras (same
            hyperparameters every fold) and returns ``PREDICTION_SCHEMA``
            rows for its test cameras.
        iou_threshold: Minimum IoU for a presence hit.
        n_val_cameras: Passed to :func:`fold_split`.

    Returns:
        See :func:`evaluate_cv`.
    """
    folds = sorted(cv_folds["fold"].unique().to_list())
    predictions = {
        fold: train_and_predict(
            fold, fold_split(cv_folds, fold, n_val_cameras=n_val_cameras)
        )
        for fold in folds
    }
    return evaluate_cv(truth, cv_folds, predictions, iou_threshold)
