"""Chunk-friendly extensions of the Detector / BehaviorClassifier protocols.

Inference keeps frames in memory and processes long videos in chunks, so it
needs detectors that carry tracker state across chunks and classifiers that
take in-memory frames. Both still satisfy the base protocols in
:mod:`bird_cv.detection.types` and :mod:`bird_cv.classification.types`.
"""

from collections.abc import Iterable, Sequence
from typing import ClassVar, Literal, Protocol

import numpy as np

from bird_cv.classification.types import BehaviorClassifier, FramePrediction
from bird_cv.detection.types import Detection, Detector
from bird_cv.inference.cages import Keyframe

ClassifierInput = Literal["bbox_crop", "cage"]
"""Media a classifier takes: 224 px bird crops, or whole cage frames."""


class FrameTracker(Protocol):
    """Tracks one stream (one cage) chunk by chunk, keeping tracker state."""

    def track(self, frames: Iterable[np.ndarray], frame_offset: int) -> list[Detection]:
        """Track the next BGR frames; ``frame_offset`` is ``frames[0]``'s frame."""
        ...


class StreamingDetector(Detector, Protocol):
    """A :class:`Detector` that can also track in-memory frame chunks."""

    def frame_tracker(self) -> FrameTracker:
        """A fresh tracker for one stream."""
        ...


class FrameClassifier(BehaviorClassifier, Protocol):
    """A :class:`BehaviorClassifier` that also classifies in-memory frames."""

    input_type: ClassVar[ClassifierInput]
    num_frames: int

    def predict_frames(self, frames: Sequence[np.ndarray]) -> list[FramePrediction]:
        """One prediction per RGB frame, ``frame`` being its position in ``frames``."""
        ...


class CageSegmenter(Protocol):
    """Splits chunks of a standardized video into cages."""

    def segment_chunk(self, frames: Sequence[np.ndarray], start: int) -> list[Keyframe]:
        """Keyframe masks for BGR ``frames``, the chunk starting at canonical ``start``."""
        ...
