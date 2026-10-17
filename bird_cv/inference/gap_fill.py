"""Per-frame boxes for one cage: confidence filter, short-gap fill, ``no_bird``.

See ``docs/adr/0003-no-bird-class.md``: a frame with no usable detection
(gap longer than the carry-forward limit, or empty cage mask) is ``no_bird``.
"""

from collections.abc import Collection, Mapping, Sequence

import polars as pl

from bird_cv.classification.types import FramePrediction
from bird_cv.core.labels import NO_BIRD, validate_class
from bird_cv.core.schemas import PREDICTION_SCHEMA
from bird_cv.detection.types import Detection


def fill_gaps(
    detections: Sequence[Detection],
    n_frames: int,
    det_conf: float,
    max_gap: int,
    empty_frames: Collection[int] = (),
) -> list[Detection | None]:
    """Choose one box per canonical frame of a cage, or None for ``no_bird``.

    Detections below ``det_conf`` are dropped and the most confident
    remaining one per frame is kept. A run of at most ``max_gap`` frames
    without a box, between two frames with one, takes the box (and
    ``conf``) of the frame before it; longer gaps, and leading or trailing
    frames with no box on one side, stay None. Frames in ``empty_frames``
    (empty cage mask) are always None.

    Args:
        detections: The cage's detections, ``frame`` being canonical.
        n_frames: Number of canonical frames in the video.
        det_conf: Minimum detection confidence.
        max_gap: Longest gap, in frames, to carry the previous box through.
        empty_frames: Canonical frames where the cage mask is empty.

    Returns:
        One entry per canonical frame: the frame's box (a gap-filled entry
        is the source detection), or None.
    """
    boxes: list[Detection | None] = [None] * n_frames
    for det in detections:
        if det.conf < det_conf or not 0 <= det.frame < n_frames:
            continue
        current = boxes[det.frame]
        if current is None or det.conf > current.conf:
            boxes[det.frame] = det
    for frame in empty_frames:
        if 0 <= frame < n_frames:
            boxes[frame] = None

    boxed = [f for f, box in enumerate(boxes) if box is not None]
    for prev, nxt in zip(boxed, boxed[1:]):
        if 1 < nxt - prev <= max_gap + 1:
            for frame in range(prev + 1, nxt):
                boxes[frame] = boxes[prev]
    for frame in empty_frames:
        if 0 <= frame < n_frames:
            boxes[frame] = None
    return boxes


def cage_predictions(
    camera_id: str,
    video_id: str,
    cage_id: str,
    boxes: Sequence[Detection | None],
    labels: Mapping[int, FramePrediction],
) -> pl.DataFrame:
    """Build the :data:`PREDICTION_SCHEMA` rows of one cage, one per canonical frame.

    Args:
        camera_id: Camera id (from :mod:`bird_cv.core.ids`).
        video_id: Video id (from :mod:`bird_cv.core.ids`).
        cage_id: Cage id (from :mod:`bird_cv.core.ids`).
        boxes: Output of :func:`fill_gaps`.
        labels: Classifier prediction per boxed canonical frame. A boxed
            frame missing here or labeled None (no window covered it) is
            ``no_bird``.

    Returns:
        One row per canonical frame, in frame order.
    """
    rows = []
    for frame, box in enumerate(boxes):
        pred = labels.get(frame) if box is not None else None
        if box is None or pred is None or pred.label is None:
            rows.append(
                {
                    "camera_id": camera_id,
                    "video_id": video_id,
                    "cage_id": cage_id,
                    "frame": frame,
                    "label": NO_BIRD,
                }
            )
            continue
        rows.append(
            {
                "camera_id": camera_id,
                "video_id": video_id,
                "cage_id": cage_id,
                "frame": frame,
                "label": validate_class(pred.label),
                "confidence": pred.confidence,
                "bbox_left": box.left,
                "bbox_top": box.top,
                "bbox_width": box.width,
                "bbox_height": box.height,
                "det_conf": box.conf,
            }
        )
    return pl.DataFrame(rows, schema=PREDICTION_SCHEMA)
