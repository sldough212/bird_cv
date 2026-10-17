"""End-to-end evaluation: predictions joined to ground truth on canonical frames.

The headline metric is frame-level macro-F1 over the nine classes (ADR
0005), computed on ``GROUND_TRUTH_SCHEMA`` rows left-joined to
``PREDICTION_SCHEMA`` rows on ``KEY_COLUMNS``. Labeling holes are excluded
and counted; a ground-truth frame with no prediction row counts as a miss.
The same join gives detection presence and MOT metrics, since ground truth
and predictions carry boxes in the same convention.
"""

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.schemas import GROUND_TRUTH_SCHEMA, KEY_COLUMNS, PREDICTION_SCHEMA
from bird_cv.evaluation.metrics import (
    box_columns,
    ClassificationMetrics,
    detection_presence_metrics,
    frame_metrics,
    mot_metrics,
    per_camera_frame_metrics,
)


def load_predictions(
    path: Path, skip_files: tuple[str, ...] = ("timings.parquet",)
) -> pl.DataFrame:
    """Read prediction parquet(s) into one ``PREDICTION_SCHEMA`` frame.

    Args:
        path: A prediction parquet, or a directory searched recursively for
            them.
        skip_files: File names in the directory that aren't predictions.
            Defaults to ``("timings.parquet",)``.

    Returns:
        All predictions, cast to ``PREDICTION_SCHEMA``.
    """
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*.parquet") if p.name not in skip_files)
    )
    frames = [
        pl.read_parquet(f, columns=list(PREDICTION_SCHEMA)).cast(PREDICTION_SCHEMA)  # type: ignore[arg-type]
        for f in files
    ]
    if not frames:
        return pl.DataFrame(schema=PREDICTION_SCHEMA)
    return pl.concat(frames)


def load_ground_truth(path: Path) -> pl.DataFrame:
    """Read ground-truth parquet(s) into one ``GROUND_TRUTH_SCHEMA`` frame.

    Args:
        path: A ground-truth parquet, or a directory searched recursively
            for them.

    Returns:
        All ground-truth rows, cast to ``GROUND_TRUTH_SCHEMA``.

    Raises:
        FileNotFoundError: If ``path`` holds no parquet files.
    """
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ground-truth parquet under {path}")
    return pl.concat(
        pl.read_parquet(f, columns=list(GROUND_TRUTH_SCHEMA)).cast(GROUND_TRUTH_SCHEMA)  # type: ignore[arg-type]
        for f in files
    )


def join_predictions(
    truth: pl.DataFrame, pred: pl.DataFrame, prefix: str = "pred_"
) -> pl.DataFrame:
    """Left-join predictions onto ground truth on ``KEY_COLUMNS``.

    Args:
        truth: ``GROUND_TRUTH_SCHEMA`` rows; these define the scored frames.
        pred: ``PREDICTION_SCHEMA`` rows.
        prefix: Prefix for prediction columns. Defaults to ``"pred_"``.

    Returns:
        ``truth`` plus every prediction column prefixed with ``prefix``
        (null where a frame has no prediction).

    Raises:
        ValueError: If ``pred`` has more than one row per key.
    """
    if pred.select(KEY_COLUMNS).is_duplicated().any():
        raise ValueError("Predictions have duplicate rows for some KEY_COLUMNS")
    renamed = pred.rename(
        {c: f"{prefix}{c}" for c in pred.columns if c not in KEY_COLUMNS}
    )
    return truth.join(renamed, on=list(KEY_COLUMNS), how="left")


@dataclass(frozen=True)
class E2EResult:
    """End-to-end scores on one set of ground-truth frames.

    Attributes:
        frame: Pooled frame-level classification scores (headline macro-F1).
        per_camera: Frame macro-F1 per camera.
        detection: Presence precision/recall/F1 and mean IoU.
        mot: MOT metrics per cage plus a pooled ``"all"`` row.
        n_frames_without_prediction: Ground-truth frames with no
            prediction row (scored as misses).
    """

    frame: ClassificationMetrics
    per_camera: pl.DataFrame
    detection: dict[str, float | int]
    mot: pl.DataFrame
    n_frames_without_prediction: int

    def summary(self) -> dict[str, Any]:
        """Scalar headline numbers for logging, JSON and sweep tables."""
        mot = {"mota": math.nan, "idf1": math.nan}
        if "camera_id" in self.mot.columns:
            pooled_mot = self.mot.filter(pl.col("camera_id") == "all")
            if pooled_mot.height:
                mot = {k: pooled_mot[k].item() for k in mot}
        return (
            self.frame.summary()
            | {"n_frames_without_prediction": self.n_frames_without_prediction}
            | {f"presence_{k}": v for k, v in self.detection.items()}
            | mot
        )


def evaluate_e2e(
    truth: pl.DataFrame,
    pred: pl.DataFrame,
    iou_threshold: float = 0.5,
    with_mot: bool = True,
) -> E2EResult:
    """Score end-to-end predictions against ground truth.

    Args:
        truth: ``GROUND_TRUTH_SCHEMA`` rows to score (already restricted to
            the cameras being evaluated).
        pred: ``PREDICTION_SCHEMA`` rows; rows for frames not in ``truth``
            are ignored.
        iou_threshold: Minimum IoU for a presence hit. Defaults to 0.5.
        with_mot: Also compute MOT metrics (slower). Defaults to ``True``.

    Returns:
        The scores, see :class:`E2EResult`.
    """
    prefix = "pred_"
    joined = join_predictions(truth, pred, prefix=prefix)
    pred_boxes = joined.select(
        *KEY_COLUMNS, *[pl.col(f"{prefix}{c}").alias(c) for c in box_columns()]
    )
    pred_label = f"{prefix}label"
    return E2EResult(
        frame=frame_metrics(joined, pred_col=pred_label),
        per_camera=per_camera_frame_metrics(joined, pred_col=pred_label),
        detection=detection_presence_metrics(
            truth, pred_boxes, iou_threshold=iou_threshold
        ),
        mot=mot_metrics(truth, pred_boxes) if with_mot else pl.DataFrame(),
        n_frames_without_prediction=int(joined[pred_label].null_count()),
    )


def write_e2e_result(result: E2EResult, output_dir: Path) -> None:
    """Write an end-to-end result to a directory.

    Args:
        result: The scores to write.
        output_dir: Created if missing. Receives ``metrics.json``
            (:meth:`E2EResult.summary`), ``per_class.parquet``,
            ``confusion.parquet`` (rows true, one column per predicted
            class), ``per_camera.parquet`` and ``mot.parquet``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result.summary(), indent=2, default=float)
    )
    result.frame.per_class.write_parquet(output_dir / "per_class.parquet")
    pl.DataFrame(
        {"label": list(result.frame.classes)}
        | {c: result.frame.confusion[:, j] for j, c in enumerate(result.frame.classes)}
    ).write_parquet(output_dir / "confusion.parquet")
    result.per_camera.write_parquet(output_dir / "per_camera.parquet")
    result.mot.write_parquet(output_dir / "mot.parquet")


def sweep_settings(
    gap_fill_frames: Iterable[int], det_conf: Iterable[float]
) -> list[tuple[int, float]]:
    """Every (gap fill, detection confidence) pair of the val sweep grid."""
    return [(g, c) for g in gap_fill_frames for c in det_conf]


def run_sweep(
    truth: pl.DataFrame,
    predictions_by_setting: Mapping[tuple[int, float], pl.DataFrame],
    iou_threshold: float = 0.5,
) -> pl.DataFrame:
    """Score each (gap fill, det conf) setting on val, best first.

    The pipeline's gap-fill length and detection threshold are chosen on
    validation end-to-end macro-F1 (ADR 0005).

    Args:
        truth: Validation ``GROUND_TRUTH_SCHEMA`` rows.
        predictions_by_setting: Predictions for each
            ``(gap_fill_frames, det_conf)`` setting.
        iou_threshold: Minimum IoU for a presence hit.

    Returns:
        One row per setting: ``gap_fill_frames``, ``det_conf`` and the
        :meth:`E2EResult.summary` fields, sorted by ``macro_f1`` descending.
    """
    rows = []
    for (gap, conf), pred in predictions_by_setting.items():
        result = evaluate_e2e(truth, pred, iou_threshold, with_mot=False)
        rows.append({"gap_fill_frames": gap, "det_conf": conf} | result.summary())
    return pl.DataFrame(rows, nan_to_null=True).sort(
        "macro_f1", descending=True, nulls_last=True
    )
