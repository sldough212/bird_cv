from collections.abc import Callable
from pathlib import Path

import polars as pl
from ultralytics import YOLO

from bird_cv.evaluation.metrics import box_columns, detection_presence_metrics


def _yolo_label_box(label_file: Path) -> tuple[float, ...] | None:
    """First box of a YOLO label file as normalized (left, top, width, height)."""
    if not label_file.exists():
        return None
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cx, cy, w, h = map(float, parts[1:5])
            return (cx - w / 2, cy - h / 2, w, h)
    return None


def _boxes_frame(rows: list[tuple[str, tuple[float, ...] | None]]) -> pl.DataFrame:
    data: dict[str, list] = {"image": [r[0] for r in rows]}
    for i, col in enumerate(box_columns()):
        data[col] = [r[1][i] if r[1] else None for r in rows]
    return pl.DataFrame(
        data, schema={"image": pl.String} | {c: pl.Float64 for c in box_columns()}
    )


Box = tuple[float, float, float, float]
"""Normalized (left, top, width, height)."""

PredictBoxes = Callable[[list[Path]], list[Box | None]]
"""Each image's top box, or None if no bird."""


def split_boxes(
    dataset_root: Path,
    predict: PredictBoxes,
    split: str = "val",
    batch: int = 32,
    image_suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ground-truth and predicted boxes for every image of a YOLO-format split.

    Images are read recursively from ``dataset_root/images/{split}`` and
    labels from the mirrored path under ``dataset_root/labels/{split}``; an
    empty or missing label file means no bird. ``predict`` keeps at most one
    box per image (a cage holds at most one bird).

    Args:
        dataset_root: YOLO dataset root containing ``images/`` and ``labels/``.
        predict: Any detector's top box per image, ``batch`` images a call.
        split: Split subdirectory. Defaults to ``"val"``.
        batch: Images per predict call.
        image_suffixes: Lower-case file suffixes counted as images.

    Returns:
        ``(truth, pred)``: one row per image keyed by ``image`` (path
        relative to the split directory) with
        :func:`bird_cv.evaluation.metrics.box_columns`, null when no box.
    """
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in image_suffixes
    )
    truth_rows: list[tuple[str, tuple[float, ...] | None]] = []
    pred_rows: list[tuple[str, tuple[float, ...] | None]] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        for image, box in zip(chunk, predict(chunk), strict=True):
            rel = image.relative_to(images_dir)
            label_file = (labels_dir / rel).with_suffix(".txt")
            truth_rows.append((str(rel), _yolo_label_box(label_file)))
            pred_rows.append((str(rel), box))
    return _boxes_frame(truth_rows), _boxes_frame(pred_rows)


def split_presence_metrics(
    dataset_root: Path,
    predict: PredictBoxes,
    split: str = "val",
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Bird presence precision/recall/F1 and mean IoU of any detector on a split.

    Detectors are selected on validation presence F1 (ADR 0005).

    Returns:
        See :func:`bird_cv.evaluation.metrics.detection_presence_metrics`.
    """
    truth, pred = split_boxes(dataset_root, predict, split)
    return detection_presence_metrics(
        truth, pred, iou_threshold=iou_threshold, on=("image",)
    )


def _yolo_predict(
    model_path: str | Path, conf: float, device: int | str | None
) -> PredictBoxes:
    """A YOLO checkpoint's top box per image."""
    model = YOLO(model_path)

    def predict(images: list[Path]) -> list[Box | None]:
        results = model.predict(
            source=[str(p) for p in images],
            conf=conf,
            max_det=1,
            device=device,
            verbose=False,
        )
        boxes: list[Box | None] = []
        for result in results:
            box: Box | None = None
            if result.boxes is not None and len(result.boxes):
                cx, cy, w, h = result.boxes.xywhn[0].tolist()
                box = (cx - w / 2, cy - h / 2, w, h)
            boxes.append(box)
        return boxes

    return predict


def yolo_presence_metrics(
    dataset_root: Path,
    model_path: str | Path,
    split: str = "val",
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    device: int | str | None = None,
) -> dict[str, float | int]:
    """Bird presence precision/recall/F1 and mean IoU of a YOLO model on a split.

    Used to select YOLO checkpoints on validation presence F1 (ADR 0005).

    Args:
        dataset_root: YOLO dataset root containing ``images/`` and ``labels/``.
        model_path: YOLO weights to evaluate.
        split: Split subdirectory. Defaults to ``"val"``.
        conf: Detection confidence threshold. Defaults to 0.25.
        iou_threshold: Minimum IoU for a hit. Defaults to 0.5.
        device: Device passed to ``model.predict``.

    Returns:
        See :func:`bird_cv.evaluation.metrics.detection_presence_metrics`.
    """
    return split_presence_metrics(
        dataset_root,
        _yolo_predict(model_path, conf, device),
        split=split,
        iou_threshold=iou_threshold,
    )
