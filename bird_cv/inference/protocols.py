"""The detector and classifier interfaces inference uses.

Inference keeps frames in memory and processes long videos in chunks, so it
needs detectors that carry tracker state across chunks and classifiers that
take in-memory frames.
"""

from collections.abc import Iterable, Sequence
from typing import Literal, Protocol

import numpy as np

from bird_cv.classification.types import FramePrediction
from bird_cv.detection.types import Detection
from bird_cv.inference.cages import Keyframe

ClassifierInput = Literal["bbox_crop", "cage"]
"""Media a classifier takes: 224 px bird crops, or whole cage frames."""


class FrameTracker(Protocol):
    """Tracks one stream (one cage) chunk by chunk, keeping tracker state."""

    def track(self, frames: Iterable[np.ndarray], frame_offset: int) -> list[Detection]:
        """Track the next BGR frames; ``frame_offset`` is ``frames[0]``'s frame."""
        ...


class StreamingDetector(Protocol):
    """A detector that tracks in-memory frame chunks."""

    def frame_tracker(self) -> FrameTracker:
        """A fresh tracker for one stream."""
        ...


class FrameClassifier(Protocol):
    """A behavior classifier of in-memory frames."""

    @property
    def input_type(self) -> ClassifierInput:
        """The media it takes."""
        ...

    num_frames: int
    # Side in px cage frames are squared to, for ``cage`` input
    input_size: int

    def predict_frames(self, frames: Sequence[np.ndarray]) -> list[FramePrediction]:
        """One prediction per RGB frame, ``frame`` being its position in ``frames``."""
        ...


class CageSegmenter(Protocol):
    """Splits chunks of a standardized video into cages."""

    def segment_chunk(self, frames: Sequence[np.ndarray], start: int) -> list[Keyframe]:
        """Keyframe masks for BGR ``frames``, the chunk starting at canonical ``start``."""
        ...
