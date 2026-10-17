"""Frame, window and detection metrics (see ``docs/adr/0005-metrics-and-objectives.md``).

Classification metrics work on class names, never on model indices, so
ground truth, end-to-end predictions and VideoMAE windows are all scored by
the same code. Per-class F1 is ``2TP / (2TP + FP + FN)``; a class that
appears in neither truth nor predictions has no F1 and is left out of the
macro average, rather than counted as 0. Ground-truth frames with a null
label are labeling holes (ADR 0003): they are excluded and counted.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from bird_cv.core.labels import CLASSES
from bird_cv.core.schemas import KEY_COLUMNS, PREDICTION_SCHEMA
from bird_cv.detection.mot_evaluate import compute_mot_metrics


def box_columns() -> tuple[str, ...]:
    """The ``bbox_`` columns of ``PREDICTION_SCHEMA``, in schema order.

    Normalized top-left + size (left, top, width, height); a null
    ``bbox_left`` means no box on that frame.
    """
    return tuple(c for c in PREDICTION_SCHEMA if c.startswith("bbox_"))


@dataclass(frozen=True)
class ClassificationMetrics:
    """Scores for one set of (truth, prediction) class pairs.

    Attributes:
        macro_f1: Mean F1 over the classes present in truth or predictions.
            NaN if nothing was scored.
        per_class: One row per class: ``label``, ``precision``, ``recall``,
            ``f1`` (null where undefined) and ``support`` (true count).
        confusion: ``len(classes)`` square matrix, rows true, columns
            predicted, in ``classes`` order.
        classes: Class order of ``per_class`` and ``confusion``.
        n_scored: Pairs with a non-null truth label.
        n_excluded: Pairs dropped because the truth label is null.
        n_unknown_predictions: Scored pairs whose prediction is missing or
            not in ``classes``; they count as misses of the true class.
    """

    macro_f1: float
    per_class: pl.DataFrame
    confusion: np.ndarray
    classes: tuple[str, ...]
    n_scored: int
    n_excluded: int
    n_unknown_predictions: int

    @property
    def classes_scored(self) -> list[str]:
        """Classes whose F1 is defined, i.e. those in the macro average."""
        return self.per_class.filter(pl.col("f1").is_not_null())["label"].to_list()

    def summary(self) -> dict[str, Any]:
        """Headline fields for logging and JSON."""
        return {
            "macro_f1": self.macro_f1,
            "n_classes_scored": len(self.classes_scored),
            "classes_scored": self.classes_scored,
            "n_scored": self.n_scored,
            "n_excluded": self.n_excluded,
            "n_unknown_predictions": self.n_unknown_predictions,
        }


def classification_metrics(
    y_true: Sequence[str | None],
    y_pred: Sequence[str | None],
    classes: Sequence[str] = CLASSES,
) -> ClassificationMetrics:
    """Macro-F1, per-class scores and confusion matrix over named classes.

    Args:
        y_true: True class per item; ``None`` marks a labeling hole, which
            is excluded and counted.
        y_pred: Predicted class per item, aligned with ``y_true``. ``None``
            or a name outside ``classes`` is scored as a miss.
        classes: Class vocabulary and report order. Defaults to the nine
            frame classes.

    Returns:
        The scores, see :class:`ClassificationMetrics`.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    classes = tuple(classes)
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    confusion = np.zeros((k, k), dtype=np.int64)
    # Missed true labels whose prediction falls outside the vocabulary.
    unknown_fn = np.zeros(k, dtype=np.int64)
    n_excluded = 0
    for t, p in zip(y_true, y_pred):
        if t is None:
            n_excluded += 1
            continue
        if t not in index:
            raise ValueError(f"Unknown true class {t!r}; expected one of {classes}")
        if p in index:
            confusion[index[t], index[p]] += 1
        else:
            unknown_fn[index[t]] += 1

    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp + unknown_fn
    support = tp + fn

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)

    defined = ~np.isnan(f1)
    macro_f1 = float(f1[defined].mean()) if defined.any() else float("nan")
    per_class = pl.DataFrame(
        {
            "label": list(classes),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        nan_to_null=True,
    )
    return ClassificationMetrics(
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=confusion,
        classes=classes,
        n_scored=int(support.sum()),
        n_excluded=n_excluded,
        n_unknown_predictions=int(unknown_fn.sum()),
    )


def window_macro_f1(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> float:
    """Window-level macro-F1 for VideoMAE clips (the selection metric).

    Args:
        y_true: True behavior per clip.
        y_pred: Predicted behavior per clip.
        classes: The model's behavior vocabulary.

    Returns:
        Macro-F1 over the classes present in truth or predictions.
    """
    return classification_metrics(y_true, y_pred, classes).macro_f1


def frame_metrics(
    joined: pl.DataFrame,
    classes: Sequence[str] = CLASSES,
    truth_col: str = "label",
    pred_col: str = "pred_label",
) -> ClassificationMetrics:
    """Frame-level classification metrics on a truth/prediction join.

    Args:
        joined: One row per canonical frame and cage with the true class in
            ``truth_col`` (null for a labeling hole) and the predicted class
            in ``pred_col`` (null if the frame has no prediction).
        classes: Class vocabulary. Defaults to the nine frame classes.
        truth_col: Column holding the true class.
        pred_col: Column holding the predicted class.

    Returns:
        Pooled scores over every row of ``joined``.
    """
    return classification_metrics(
        joined[truth_col].to_list(), joined[pred_col].to_list(), classes
    )


def per_camera_frame_metrics(
    joined: pl.DataFrame,
    classes: Sequence[str] = CLASSES,
    truth_col: str = "label",
    pred_col: str = "pred_label",
) -> pl.DataFrame:
    """Frame macro-F1 for each camera of a truth/prediction join.

    Args:
        joined: As for :func:`frame_metrics`, plus a ``camera_id`` column.
        classes: Class vocabulary.
        truth_col: Column holding the true class.
        pred_col: Column holding the predicted class.

    Returns:
        One row per camera, sorted: ``camera_id`` and the fields of
        :meth:`ClassificationMetrics.summary`.
    """
    rows = []
    for (camera_id,), group in joined.sort("camera_id").group_by(
        "camera_id", maintain_order=True
    ):
        m = frame_metrics(group, classes, truth_col, pred_col)
        rows.append({"camera_id": camera_id} | m.summary())
    return pl.DataFrame(
        rows,
        schema={
            "camera_id": pl.String,
            "macro_f1": pl.Float64,
            "n_classes_scored": pl.Int64,
            "classes_scored": pl.List(pl.String),
            "n_scored": pl.Int64,
            "n_excluded": pl.Int64,
            "n_unknown_predictions": pl.Int64,
        },
        nan_to_null=True,
    )


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of two aligned box arrays.

    Args:
        a: ``(N, 4)`` boxes as ``left, top, width, height``.
        b: ``(N, 4)`` boxes in the same convention, row ``i`` compared
            with row ``i`` of ``a``.

    Returns:
        ``(N,)`` IoU per row; 0 where both boxes have zero area.
    """
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _with_presence(df: pl.DataFrame, on: Sequence[str], prefix: str) -> pl.DataFrame:
    return df.select(
        *on, *[pl.col(c).alias(f"{prefix}{c}") for c in box_columns()]
    ).unique(subset=list(on), keep="first")


def detection_presence_metrics(
    truth: pl.DataFrame,
    pred: pl.DataFrame,
    iou_threshold: float = 0.5,
    on: Sequence[str] = KEY_COLUMNS,
) -> dict[str, float | int]:
    """Per-frame bird presence precision/recall and mean IoU on matches.

    Each frame has at most one true and one predicted box (a cage holds at
    most one bird). A frame is a true positive when both have a box with
    IoU ``>= iou_threshold``; a predicted box with lower IoU is both a false
    positive and a miss. Frames come from ``truth``; a frame missing from
    ``pred`` has no predicted box.

    Args:
        truth: ``on`` columns plus :func:`box_columns`; null ``bbox_left``
            means no bird.
        pred: Same columns for the predictions.
        iou_threshold: Minimum IoU for a hit. Defaults to 0.5.
        on: Columns identifying one frame. Defaults to ``KEY_COLUMNS``.

    Returns:
        ``tp``, ``fp``, ``fn``, ``tn``, ``precision``, ``recall``, ``f1``
        and ``mean_iou`` (over true positives; NaN if there are none).
    """
    joined = _with_presence(truth, on, "t_").join(
        _with_presence(pred, on, "p_"), on=list(on), how="left"
    )
    has_t = joined["t_bbox_left"].is_not_null().to_numpy()
    has_p = joined["p_bbox_left"].is_not_null().to_numpy()
    both = has_t & has_p

    t_boxes = joined.select([f"t_{c}" for c in box_columns()]).to_numpy()[both]
    p_boxes = joined.select([f"p_{c}" for c in box_columns()]).to_numpy()[both]
    ious = box_iou(t_boxes.astype(float), p_boxes.astype(float))
    hit = ious >= iou_threshold

    tp = int(hit.sum())
    fp = int((has_p & ~has_t).sum() + (~hit).sum())
    fn = int((has_t & ~has_p).sum() + (~hit).sum())
    tn = int((~has_t & ~has_p).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(ious[hit].mean()) if tp else float("nan"),
    }


def _to_mot(df: pl.DataFrame, frame_offset: int, track_id: int) -> np.ndarray:
    """Rows with a box as a MOT array (frame, id, left, top, w, h, conf, -1...)."""
    boxed = df.filter(pl.col("bbox_left").is_not_null()).sort("frame")
    n = boxed.height
    if n == 0:
        return np.empty((0, 10), dtype=float)
    out = np.full((n, 10), -1.0)
    out[:, 0] = boxed["frame"].to_numpy() + frame_offset
    out[:, 1] = track_id
    out[:, 2:6] = boxed.select(box_columns()).to_numpy()
    out[:, 6] = 1.0
    return out


def mot_metrics(truth: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """MOTA, IDF1 etc. per cage and pooled, via :func:`compute_mot_metrics`.

    Reported, not optimized (ADR 0005). Each cage has one true track; each
    cage's predictions are one track too, since predictions carry no track
    id and a cage holds at most one bird. The pooled row (``camera_id ==
    "all"``) runs every cage through one accumulator, with frames offset
    so cages never share a frame and ids unique per cage, which gives exact
    pooled counts rather than an average of per-cage ratios.

    Args:
        truth: ``KEY_COLUMNS`` plus :func:`box_columns`; null ``bbox_left``
            means no bird.
        pred: Same columns for the predictions.

    Returns:
        One row per cage plus the pooled row: ``camera_id``, ``video_id``,
        ``cage_id`` and one column per metric.
    """
    cage_columns = ["camera_id", "video_id", "cage_id"]
    cages = truth.select(cage_columns).unique().sort(cage_columns).rows()
    records = []
    gt_all, pred_all = [], []
    offset = 0
    for i, (camera_id, video_id, cage_id) in enumerate(cages, start=1):
        cage = (
            (pl.col("camera_id") == camera_id)
            & (pl.col("video_id") == video_id)
            & (pl.col("cage_id") == cage_id)
        )
        t, p = truth.filter(cage), pred.filter(cage)
        records.append(
            {"camera_id": camera_id, "video_id": video_id, "cage_id": cage_id}
            | compute_mot_metrics(gt=_to_mot(t, 0, 1), pred=_to_mot(p, 0, 1))
        )
        gt_all.append(_to_mot(t, offset, i))
        pred_all.append(_to_mot(p, offset, i))
        max_frame = max(
            t["frame"].max() if t.height else 0, p["frame"].max() if p.height else 0
        )
        offset += int(max_frame) + 1  # type: ignore[call-overload]

    if cages:
        records.append(
            {"camera_id": "all", "video_id": "all", "cage_id": "all"}
            | compute_mot_metrics(gt=np.vstack(gt_all), pred=np.vstack(pred_all))
        )
    return pl.DataFrame(records)
