"""Behavior classification over contiguous detected runs of one cage.

Mirrors how Classify training clips are built from bird-crop videos (which
hold only boxed frames): the classifier only ever sees boxed frames, windows
never span a gap, and a run shorter than one window is centered and padded
with neighboring boxed frames via ``annotations_to_mae._center_window``.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from bird_cv.classification.types import FramePrediction
from bird_cv.detection.types import Detection
from bird_cv.inference.protocols import FrameClassifier
from bird_cv.preprocessing.annotations_to_mae import _center_window
from bird_cv.preprocessing.annotations_to_yolo import crop_bird, smooth_centers


def smoothed_box_centers(
    boxes: Sequence[Detection | None],
) -> dict[int, tuple[float, float]]:
    """Smoothed normalized box center per boxed canonical frame of a cage.

    Centers are smoothed across the cage's boxed frames in order, as the
    Classify bird-crop videos are.
    """
    frames = [f for f, box in enumerate(boxes) if box is not None]
    if not frames:
        return {}
    cx = np.array([boxes[f].left + boxes[f].width / 2 for f in frames])  # type: ignore[union-attr]
    cy = np.array([boxes[f].top + boxes[f].height / 2 for f in frames])  # type: ignore[union-attr]
    cx_s, cy_s = smooth_centers(cx, cy)
    return {f: (float(x), float(y)) for f, x, y in zip(frames, cx_s, cy_s)}


def bird_crop_rgb(
    cage_frame: np.ndarray, center: tuple[float, float], crop_size: int
) -> np.ndarray:
    """Bird crop of a BGR cage frame, converted to RGB for the classifier."""
    crop = crop_bird(cage_frame, center[0], center[1], crop_size=crop_size)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def contiguous_runs(frames: Sequence[int]) -> list[tuple[int, int]]:
    """``[start, end)`` index ranges of consecutive values in sorted ``frames``."""
    runs = []
    start = 0
    for i in range(1, len(frames) + 1):
        if i == len(frames) or frames[i] != frames[i - 1] + 1:
            if frames:
                runs.append((start, i))
            start = i
    return runs


def classify_runs(
    frames: Sequence[int],
    crops: Sequence[np.ndarray],
    classifier: FrameClassifier,
) -> dict[int, FramePrediction]:
    """Classify each contiguous run of boxed frames.

    Args:
        frames: Sorted canonical frames that have a box.
        crops: RGB bird crop of each frame in ``frames``.
        classifier: Classifier taking ``bbox_crop`` input.

    Returns:
        Prediction per canonical frame in ``frames``; ``frame`` holds the
        canonical frame.
    """
    num = classifier.num_frames
    predictions: dict[int, FramePrediction] = {}
    for start, end in contiguous_runs(frames):
        if end - start >= num:
            window = list(crops[start:end])
            offset = 0
            win_start = start
        else:
            win_start, win_end = _center_window(start, end, num, len(crops))
            window = list(crops[win_start:win_end])
            # Fewer boxed frames than one window: repeat the edge frames, as
            # annotations_to_mae does for short videos
            left_pad = (num - len(window)) // 2
            right_pad = num - len(window) - left_pad
            window = [window[0]] * left_pad + window + [window[-1]] * right_pad
            offset = left_pad
        window_preds = classifier.predict_frames(window)
        for i in range(start, end):
            pred = window_preds[i - win_start + offset]
            predictions[frames[i]] = FramePrediction(
                frame=frames[i], label=pred.label, confidence=pred.confidence
            )
    return predictions
