"""Bare-video YOLO tracking, independent of any Label Studio task/project.

Factored out of ``bird_cv.client.post_bbox_predictions.get_yolo_predictions``,
which wraps this same tracking call per Label Studio task and reshapes the
result into ``videorectangle`` prediction regions. This module skips that
wrapping entirely so a single cage video can be tracked on its own -- e.g.
as the detection step of a single-video inference pipeline, with the raw
per-frame boxes written out directly as an intermediate evaluation artifact.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from bird_cv.detection.types import Detection


def track_video(
    video_path: Path | str,
    model_path: Path | str,
    tracker_path: Path | str,
    conf: float = 0.05,
    iou: float = 0.70,
    max_det: int = 1,
) -> list[Detection]:
    """Run YOLO tracking over a single video and return per-frame boxes.

    Args:
        video_path: Path to the video to track (e.g. a cropped per-cage
            video).
        model_path: Path to the YOLO model weights (e.g. a ``.pt`` file).
        tracker_path: Path to the tracker config (e.g. ``tracker.yaml``)
            passed to ``model.track``.
        conf: Confidence threshold passed to ``model.track``. Defaults to
            0.05 to match the Label Studio prediction path.
        iou: NMS IoU threshold passed to ``model.track``. Defaults to 0.70.
        max_det: Maximum detections per frame passed to ``model.track``.
            Defaults to 1 (at most one bird per cage).

    Returns:
        One :class:`Detection` per tracked box per frame it appears in,
        in frame order. Frames with no detection contribute no entries.
    """
    model = YOLO(str(model_path))
    model_names = model.names

    results = model.track(
        str(video_path),
        stream=True,
        conf=conf,
        iou=iou,
        max_det=max_det,
        tracker=str(tracker_path),
    )

    detections: list[Detection] = []
    for frame_idx, result in enumerate(results):
        detections.extend(_result_detections(result, frame_idx, model_names))

    return detections


def _result_detections(
    result: Results, frame_idx: int, model_names: dict[int, str]
) -> list[Detection]:
    """Convert one frame's tracking result into :class:`Detection` rows.

    Untracked boxes (no ``id``) are dropped, matching the Label Studio path.
    """
    data = result.boxes
    if data is None or not data.is_track:
        return []

    detections = []
    for i, track_id in enumerate(data.id.tolist()):
        x, y, w, h = data.xywhn[i].tolist()
        detections.append(
            Detection(
                frame=frame_idx,
                track_id=int(track_id),
                left=x - w / 2,
                top=y - h / 2,
                width=w,
                height=h,
                conf=float(data.conf[i]),
                label=model_names[int(data.cls[i])],
            )
        )
    return detections


def track_frames(
    model: YOLO,
    frames: Iterable[np.ndarray],
    tracker_path: Path | str,
    conf: float = 0.05,
    iou: float = 0.70,
    max_det: int = 1,
    *,
    frame_offset: int = 0,
) -> list[Detection]:
    """Track in-memory frames, carrying tracker state across calls.

    Each frame goes through ``model.track(..., persist=True)``, so calling
    this repeatedly with the same ``model`` on consecutive chunks of one
    video continues the same tracks, as if the whole video had been passed
    to :func:`track_video`. Use a separate ``model`` per independent stream
    (e.g. one per cage).

    Args:
        model: Loaded YOLO model; owns the tracker state.
        frames: BGR frames, in order.
        tracker_path: Path to the tracker config (e.g. ``tracker.yaml``).
        conf: Confidence threshold passed to ``model.track``.
        iou: NMS IoU threshold passed to ``model.track``.
        max_det: Maximum detections per frame passed to ``model.track``.
        frame_offset: Frame number of ``frames[0]``; added to every
            returned ``Detection.frame``.

    Returns:
        One :class:`Detection` per tracked box per frame, in frame order.
    """
    detections: list[Detection] = []
    for i, frame in enumerate(frames):
        result = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            max_det=max_det,
            tracker=str(tracker_path),
            verbose=False,
        )[0]
        detections.extend(_result_detections(result, frame_offset + i, model.names))
    return detections


@dataclass
class YoloFrameTracker:
    """One YOLO model and tracker state for a single stream of frames.

    Attributes:
        model_path: Path to the YOLO model weights.
        tracker_path: Path to the tracker config.
        conf: Confidence threshold passed to ``model.track``.
        iou: NMS IoU threshold passed to ``model.track``.
        max_det: Maximum detections per frame passed to ``model.track``.
    """

    model_path: Path | str
    tracker_path: Path | str
    conf: float = 0.05
    iou: float = 0.70
    max_det: int = 1

    def __post_init__(self) -> None:
        self.model = YOLO(str(self.model_path))

    def track(self, frames: Iterable[np.ndarray], frame_offset: int) -> list[Detection]:
        """Track the next chunk of this stream; see :func:`track_frames`."""
        return track_frames(
            self.model,
            frames,
            self.tracker_path,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            frame_offset=frame_offset,
        )


@dataclass
class YoloDetector:
    """:class:`bird_cv.detection.types.Detector` backed by a YOLO checkpoint.

    Wraps :func:`track_video` so detectors are swappable: anything that
    implements ``predict(video_path) -> list[Detection]`` can stand in for
    this in the single-video pipeline and its evaluation harness.

    Attributes:
        model_path: Path to the YOLO model weights (e.g. a ``.pt`` file).
        tracker_path: Path to the tracker config (e.g. ``tracker.yaml``).
        conf: Confidence threshold passed to ``model.track``.
        iou: NMS IoU threshold passed to ``model.track``.
        max_det: Maximum detections per frame passed to ``model.track``.
    """

    model_path: Path | str
    tracker_path: Path | str
    conf: float = 0.05
    iou: float = 0.70
    max_det: int = 1

    def predict(self, video_path: Path | str) -> list[Detection]:
        return track_video(
            video_path=video_path,
            model_path=self.model_path,
            tracker_path=self.tracker_path,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
        )

    def frame_tracker(self) -> YoloFrameTracker:
        """Return a fresh tracker for one stream of in-memory frame chunks."""
        return YoloFrameTracker(
            model_path=self.model_path,
            tracker_path=self.tracker_path,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
        )
