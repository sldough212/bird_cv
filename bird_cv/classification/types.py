"""Model-agnostic data model for the behavior-classification stage.

Every classifier -- whatever VideoMAE checkpoint, or a different
architecture entirely -- takes a (bbox-cropped) video and returns
``list[FramePrediction]``. The single-video pipeline and the classification
evaluation harness only ever talk to that type, never to a specific model's
native output format, so a new classifier can be swapped in by writing a
class that implements :class:`BehaviorClassifier` and nothing downstream has
to change. The same type is also the target for parsing ground-truth
behavior labels (e.g. from exported Label Studio timelinelabels), so a
prediction and its ground truth are always directly comparable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class FramePrediction:
    """One classified frame of a behavior video.

    Attributes:
        frame: 0-indexed frame number within the video.
        label: Predicted behavior label, or ``None`` if the model made no
            prediction for this frame (e.g. no window covers it). The
            single-video pipeline overwrites this with ``"no_bird"``
            for frames that never had a bounding box to crop in the first
            place -- a fact the classifier itself has no way to know,
            since by the time it sees the video those frames are already
            gone or placeholdered.
        confidence: 0-1, model-defined. Not necessarily a calibrated
            probability -- e.g. a sliding-window voting classifier reports
            the winning label's vote share, while a single-pass classifier
            might report a softmax probability. Comparable in spirit, not
            guaranteed comparable in value, across different classifiers.
            Ground-truth labels use 1.0.
    """

    frame: int
    label: str | None
    confidence: float


class BehaviorClassifier(Protocol):
    """Anything that can turn a behavior video into per-frame labels."""

    def predict(self, video_path: Path) -> list[FramePrediction]:
        """Run behavior classification over a video.

        Args:
            video_path: Path to the video to run on (e.g. a per-bird,
                bbox-cropped video).

        Returns:
            One :class:`FramePrediction` per frame of the video, in frame
            order.
        """
        ...
