"""Bare-video YOLO tracking, independent of any Label Studio task/project.

Factored out of ``bird_cv.client.post_bbox_predictions.get_yolo_predictions``,
which wraps this same tracking call per Label Studio task and reshapes the
result into ``videorectangle`` prediction regions. This module skips that
wrapping entirely so a single cage video can be tracked on its own -- e.g.
as the detection step of a single-video inference pipeline, with the raw
per-frame boxes written out directly as an intermediate evaluation artifact.
"""

from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

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
        data = result.boxes
        if data is None or not data.is_track:
            continue

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
