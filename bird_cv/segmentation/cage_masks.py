"""Cage masks without prompts: saved keyframes, YOLO labels, static masks.

Ground truth is the keyframe masks a ``sam2`` inference run saved per video
(:meth:`bird_cv.core.layout.InferRun.segments`, next to the standardized
video they were made on). Cages never move within a video, so a trained
segmenter gives each cage one *static mask* per video, from a pixel-wise
majority vote over a few sampled frames, and numbers cages in row-major
order (ADR 0007).
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

Keyframes = list[tuple[int, dict[str, np.ndarray]]]
"""``[(first canonical frame, {cage_id: 2D bool mask})]``, in frame order."""


def save_keyframes(path: Path, keyframes: Keyframes) -> None:
    """Write keyframes as ``.npz`` arrays named ``{start}_{cage_id}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{f"{s}_{c}": m for s, cages in keyframes for c, m in cages.items()}
    )


def keyframe_starts(path: Path) -> list[int]:
    """First frames of the keyframes in a :func:`save_keyframes` file (no masks read)."""
    with np.load(path) as data:
        return sorted({int(k.split("_")[0]) for k in data.files})


def load_keyframes(path: Path) -> Keyframes:
    """Read a :func:`save_keyframes` file, in frame order."""
    keyframes: dict[int, dict[str, np.ndarray]] = {}
    with np.load(path) as data:
        for key in data.files:
            start, cage_id = key.split("_")
            keyframes.setdefault(int(start), {})[cage_id] = data[key]
    return sorted(keyframes.items())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks; 0 when both are empty."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def match_masks(
    a: list[np.ndarray], b: list[np.ndarray], min_iou: float = 0.0
) -> list[tuple[int, int, float]]:
    """Hungarian matching of two mask lists on IoU.

    Returns:
        ``(i, j, iou)`` per matched pair with ``iou > min_iou``.
    """
    if not a or not b:
        return []
    ious = np.array([[mask_iou(x, y) for y in b] for x in a])
    rows, cols = linear_sum_assignment(-ious)
    return [(i, j, ious[i, j]) for i, j in zip(rows, cols) if ious[i, j] > min_iou]


def mask_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Largest outer contour of a mask, normalized ``(x, y)`` points; None if empty."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    points = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(points) < 3:
        return None
    h, w = mask.shape
    return points / [w, h]


def yolo_seg_label(masks: list[np.ndarray]) -> str:
    """Ultralytics segmentation label text, class 0, one polygon line per mask."""
    lines = []
    for mask in masks:
        polygon = mask_polygon(mask)
        if polygon is not None:
            lines.append("0 " + " ".join(f"{v:.6f}" for v in polygon.ravel()))
    return "\n".join(lines)


def row_major_order(masks: list[np.ndarray]) -> list[int]:
    """Indices of ``masks`` top-to-bottom by row, then left-to-right.

    A row starts at the highest remaining cage and takes every cage whose
    centroid lies within that cage's vertical extent.
    """
    info = []
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        info.append((i, ys.mean(), xs.mean(), ys.min(), ys.max()))
    remaining = sorted(info, key=lambda c: c[1])
    order: list[int] = []
    while remaining:
        _, _, _, top, bottom = remaining[0]
        row = [c for c in remaining if top <= c[1] <= bottom]
        order += [c[0] for c in sorted(row, key=lambda c: c[2])]
        remaining = [c for c in remaining if c not in row]
    return order


def static_cage_masks(
    detections: list[list[tuple[np.ndarray, float]]],
    expected_cages: int | None = None,
    min_iou: float = 0.5,
) -> list[np.ndarray]:
    """One mask per cage from detections on several frames of one video.

    Cages are the detections of the frame with the most of them; every
    other frame's detections are matched to those on IoU. A cage's mask is
    the pixel-wise majority over its matched masks.

    Args:
        detections: Per sampled frame, ``(mask, confidence)`` per detection.
        expected_cages: Keep this many cages, the most confident on average;
            None keeps all.
        min_iou: IoU a detection needs to match a cage.

    Returns:
        The cage masks in row-major order (:func:`row_major_order`).

    Raises:
        ValueError: If no cage was detected, or fewer than ``expected_cages``.
    """
    reference = max(detections, key=len, default=[])
    cages = [([mask], [conf]) for mask, conf in reference]
    for frame in detections:
        if frame is reference:
            continue
        matches = match_masks([c[0][0] for c in cages], [m for m, _ in frame], min_iou)
        for i, j, _ in matches:
            cages[i][0].append(frame[j][0])
            cages[i][1].append(frame[j][1])
    found = len(cages)
    if not found or (expected_cages is not None and found < expected_cages):
        raise ValueError(f"Found {found} cages, expected {expected_cages or 'some'}")
    if expected_cages is not None:
        cages = sorted(cages, key=lambda c: -np.mean(c[1]))[:expected_cages]
    masks = [np.mean(np.stack(m), axis=0) > 0.5 for m, _ in cages]
    return [masks[i] for i in row_major_order(masks)]
