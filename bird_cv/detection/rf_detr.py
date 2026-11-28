"""RF-DETR detector: training, top-box prediction and tracking (ADR 0010).

Tracks with the same Ultralytics tracker and ``tracker.yaml`` as
:class:`~bird_cv.detection.track.YoloDetector`, so the two kinds differ only
in their detector. Trains on the YOLO-format dataset of the ``yolo`` kind.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rfdetr
import supervision as sv
from PIL import Image
from ultralytics.engine.results import Boxes
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace

from bird_cv.core.layout import TrainRun
from bird_cv.detection.evaluate_yolo import Box, PredictBoxes, split_presence_metrics
from bird_cv.detection.train_yolo import write_dataset_yaml
from bird_cv.detection.types import Detection


def top_box(detections: sv.Detections) -> np.ndarray:
    """The most confident box as ``[[x1, y1, x2, y2, conf, cls]]`` (pixels), or no rows."""
    if len(detections) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    i = int(np.argmax(detections.confidence))
    return np.array(
        [[*detections.xyxy[i], detections.confidence[i], detections.class_id[i]]],
        dtype=np.float32,
    )


def rfdetr_predict(model: Any, threshold: float) -> PredictBoxes:
    """An RF-DETR model's top box per image, normalized."""

    def predict(images: list[Path]) -> list[Box | None]:
        rgb = [Image.open(p).convert("RGB") for p in images]
        preds = model.predict(rgb, threshold=threshold, include_source_image=False)
        boxes: list[Box | None] = []
        for img, det in zip(rgb, preds, strict=True):
            row = top_box(det)
            w, h = img.size
            boxes.append(
                (
                    row[0, 0] / w,
                    row[0, 1] / h,
                    (row[0, 2] - row[0, 0]) / w,
                    (row[0, 3] - row[0, 1]) / h,
                )
                if len(row)
                else None
            )
        return boxes

    return predict


def rfdetr_presence_metrics(
    dataset_root: Path,
    checkpoint: Path,
    split: str = "val",
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """Presence metrics of an RF-DETR checkpoint on a YOLO-format split."""
    model = rfdetr.RFDETR.from_checkpoint(str(checkpoint))
    return split_presence_metrics(
        dataset_root, rfdetr_predict(model, conf), split, iou_threshold
    )


def train_rfdetr(
    dataset_root: Path,
    run_dir: Path,
    pretrained: str,
    epochs: int,
    device: int | str,
    train_options: dict[str, Any],
    select_conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Train RF-DETR once and score its best checkpoint on val presence F1.

    RF-DETR keeps its best epoch by EMA val mAP (``checkpoint_best_total.pth``);
    the score goes to ``selection.parquet`` like YOLO's candidates.

    Args:
        dataset_root: YOLO-format dataset; ``data.yaml`` is written into it.
        run_dir: Run directory, RF-DETR's ``output_dir``.
        pretrained: An ``rfdetr`` model class, e.g. ``"RFDETRSmall"``.
        epochs: Training epochs.
        device: GPU index or torch device string.
        train_options: ``model.train`` keyword arguments.
        select_conf: Detection confidence used when scoring.
        iou_threshold: Minimum IoU for a presence hit.

    Returns:
        ``checkpoint`` and its val presence metrics.
    """
    write_dataset_yaml(dataset_root, dataset_root / "data.yaml")
    model = getattr(rfdetr, pretrained)()
    model.train(
        dataset_dir=str(dataset_root),
        epochs=epochs,
        device=f"cuda:{device}" if isinstance(device, int) else device,
        output_dir=str(run_dir),
        **train_options,
    )
    checkpoint = run_dir / "checkpoint_best_total.pth"
    metrics = rfdetr_presence_metrics(
        dataset_root, checkpoint, "val", select_conf, iou_threshold
    )
    pl.DataFrame(
        [{"candidate": run_dir.name, "checkpoint": str(checkpoint)} | metrics]
    ).with_columns(
        within_run_selection=pl.lit("rfdetr_val_map"),
        between_run_selection=pl.lit("val_presence_f1"),
    ).write_parquet(TrainRun(run_dir).selection)
    return {"checkpoint": checkpoint} | metrics


class RfdetrFrameTracker:
    """Tracks one stream: RF-DETR's top box per frame through an Ultralytics tracker.

    Args:
        model: Loaded RF-DETR model, possibly shared with other trackers.
        tracker: This stream's Ultralytics tracker (e.g. ``BOTSORT``).
        threshold: RF-DETR confidence threshold; keep it under the
            tracker's ``track_low_thresh`` so its second pass gets boxes.
        batch_size: Frames per predict call.
    """

    def __init__(
        self, model: Any, tracker: Any, threshold: float, batch_size: int
    ) -> None:
        self.model = model
        self.tracker = tracker
        self.threshold = threshold
        self.batch_size = batch_size

    def track(self, frames: Iterable[np.ndarray], frame_offset: int) -> list[Detection]:
        """Track BGR frames; ``frame_offset`` is ``frames[0]``'s frame."""
        frames = list(frames)
        names = self.model.class_names
        out: list[Detection] = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start : start + self.batch_size]
            preds = self.model.predict(
                [np.ascontiguousarray(f[..., ::-1]) for f in batch],
                threshold=self.threshold,
                include_source_image=False,
            )
            for i, (frame, det) in enumerate(zip(batch, preds, strict=True)):
                h, w = frame.shape[:2]
                # Rows [x1, y1, x2, y2, track_id, score, cls, idx], Kalman boxes
                tracks = self.tracker.update(Boxes(top_box(det), (h, w)), frame)
                out.extend(
                    Detection(
                        frame=frame_offset + start + i,
                        track_id=int(t[4]),
                        left=float(t[0] / w),
                        top=float(t[1] / h),
                        width=float((t[2] - t[0]) / w),
                        height=float((t[3] - t[1]) / h),
                        conf=float(t[5]),
                        label=names[int(t[6])],
                    )
                    for t in tracks
                )
        return out


@dataclass
class RfdetrDetector:
    """:class:`~bird_cv.inference.protocols.StreamingDetector` backed by an
    RF-DETR checkpoint.

    Attributes:
        model_path: RF-DETR checkpoint (``checkpoint_best_total.pth``).
        tracker_path: Ultralytics tracker config (e.g. ``tracker.yaml``).
        threshold: RF-DETR confidence threshold.
        batch_size: Frames per predict call.
    """

    model_path: Path | str
    tracker_path: Path | str
    threshold: float = 0.05
    batch_size: int = 16
    _model: Any = field(default=None, init=False, repr=False)

    def frame_tracker(self) -> RfdetrFrameTracker:
        """A fresh tracker for one stream; all share one loaded model."""
        if self._model is None:
            self._model = rfdetr.RFDETR.from_checkpoint(str(self.model_path))
        cfg = IterableSimpleNamespace(**YAML.load(str(self.tracker_path)))
        # frame_rate=30 as in Ultralytics' model.track
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        return RfdetrFrameTracker(self._model, tracker, self.threshold, self.batch_size)
