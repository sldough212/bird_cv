"""Model-agnostic data model for the detection stage.

Every detector -- whatever YOLO checkpoint, or a different architecture
entirely -- returns ``list[Detection]`` (see
:class:`bird_cv.inference.protocols.StreamingDetector`). Inference and
evaluation only ever talk to that type, never to a specific model's native
output format, so a new detector can be swapped in without changing
anything downstream. The same type is also the target for parsing
ground-truth boxes (e.g. from exported Label Studio annotations), so a
prediction and its ground truth are always directly comparable.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One tracked bounding box in one frame of a cage video.

    Coordinates are normalized 0-1 (fraction of frame width/height),
    top-left + size. Normalized rather than pixel space because IoU --
    all this project's box matching and MOT metrics are based on it -- is
    scale invariant, so there's never a need to know the frame's pixel
    dimensions just to compare boxes.

    Attributes:
        frame: 0-indexed frame number within the video.
        track_id: Identifier distinguishing one tracked box from another
            across frames. Not required to be comparable between a
            prediction and its ground truth (MOT metrics match boxes by
            IoU, not by id).
        left: Left edge, normalized 0-1.
        top: Top edge, normalized 0-1.
        width: Box width, normalized 0-1.
        height: Box height, normalized 0-1.
        conf: Model confidence for this detection, 0-1. Ground-truth boxes
            use 1.0.
        label: Class label (e.g. ``"bird"``).
    """

    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float
    label: str
