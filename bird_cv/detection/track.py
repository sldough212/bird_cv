"""Bare-video YOLO tracking, independent of any Label Studio task/project.

Factored out of ``bird_cv.client.post_bbox_predictions.get_yolo_predictions``,
which wraps this same tracking call per Label Studio task and reshapes the
result into ``videorectangle`` prediction regions. This module skips that
wrapping entirely so a single cage video can be tracked on its own -- e.g.
as the detection step of a single-video inference pipeline, with the raw
per-frame boxes written out directly as an intermediate evaluation artifact.
"""

from pathlib import Path

import polars as pl
from ultralytics import YOLO


def track_video(
    video_path: Path | str,
    model_path: Path | str,
    tracker_path: Path | str,
    conf: float = 0.05,
    iou: float = 0.70,
    max_det: int = 1,
) -> pl.DataFrame:
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
        Polars DataFrame with one row per tracked detection and columns
        ``frame`` (0-indexed), ``track_id``, ``left``, ``top``, ``width``,
        ``height`` (all four normalized 0-1, top-left + size), ``conf``,
        and ``label``. Frames with no detection contribute no rows.
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

    rows: list[dict] = []
    for frame_idx, result in enumerate(results):
        data = result.boxes
        if data is None or not data.is_track:
            continue

        for i, track_id in enumerate(data.id.tolist()):
            x, y, w, h = data.xywhn[i].tolist()
            rows.append(
                {
                    "frame": frame_idx,
                    "track_id": int(track_id),
                    "left": x - w / 2,
                    "top": y - h / 2,
                    "width": w,
                    "height": h,
                    "conf": float(data.conf[i]),
                    "label": model_names[int(data.cls[i])],
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "frame": pl.Int64,
            "track_id": pl.Int64,
            "left": pl.Float64,
            "top": pl.Float64,
            "width": pl.Float64,
            "height": pl.Float64,
            "conf": pl.Float64,
            "label": pl.Utf8,
        },
    )
