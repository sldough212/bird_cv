"""Cage masks of an inference run against the legacy one-shot SAM2 masks.

Cages never move within a video, so each video's first predicted keyframe is
compared with every legacy keyframe of that video; no frame alignment is
needed. Cages are matched on IoU (Hungarian), and a cage only one side has
scores 0.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bird_cv.segmentation.cage_masks import load_legacy_keyframes, match_masks


def first_keyframe(segments: Path) -> list[np.ndarray]:
    """Cage masks of the first keyframe in an inference ``segments`` file."""
    with np.load(segments) as data:
        keys = [k.split("_") for k in data.files]
        start = min(int(s) for s, _ in keys)
        return [data[f"{s}_{c}"] for s, c in keys if int(s) == start]


def _fit(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """``mask`` cut or zero-padded bottom/right to ``shape`` (standardizing pads odd sides)."""
    out = np.zeros(shape, dtype=bool)
    h, w = min(shape[0], mask.shape[0]), min(shape[1], mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


def compare_masks(pred: list[np.ndarray], truth: list[np.ndarray]) -> dict[str, Any]:
    """Matched IoU of predicted and true cages of one frame.

    Returns:
        ``truth_cages``, ``pred_cages``, ``mean_iou`` and ``min_iou`` over
        every cage of either side (unmatched cages score 0).
    """
    pred = [_fit(m, truth[0].shape) for m in pred] if truth else pred
    ious = [iou for _, _, iou in match_masks(pred, truth)]
    ious += [0.0] * (max(len(pred), len(truth)) - len(ious))
    return {
        "truth_cages": len(truth),
        "pred_cages": len(pred),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
    }


def evaluate_segments(
    segments_dir: Path, masks_root: Path, cameras: set[str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score every video of ``cameras`` an inference run segmented.

    Args:
        segments_dir: An inference run's ``segments/`` (``{camera}/{video}.npz``).
        masks_root: Legacy masks, ``{camera}/{video}/``.
        cameras: Cameras to score (e.g. the test split).

    Returns:
        One row per (video, legacy segment), and the summary: ``videos``,
        ``mean_iou``, ``worst_cage_iou`` (mean of per-segment minimums) and
        ``count_accuracy`` (segments with the true cage count).

    Raises:
        FileNotFoundError: If no video was scored, or a video has no legacy masks.
    """
    rows = []
    for path in sorted(segments_dir.glob("*/*.npz")):
        camera_id, video_id = path.parent.name, path.stem
        if camera_id not in cameras:
            continue
        pred = first_keyframe(path)
        for seg, (_, truth) in enumerate(
            load_legacy_keyframes(masks_root / camera_id / video_id)
        ):
            rows.append(
                {"camera_id": camera_id, "video_id": video_id, "segment": seg}
                | compare_masks(pred, list(truth.values()))
            )
    if not rows:
        raise FileNotFoundError(
            f"No segmented videos of {sorted(cameras)} in {segments_dir}"
        )
    table = pl.DataFrame(rows)
    summary = {
        "videos": table.select("camera_id", "video_id").n_unique(),
        "mean_iou": table["mean_iou"].mean(),
        "worst_cage_iou": table["min_iou"].mean(),
        "count_accuracy": (table["truth_cages"] == table["pred_cages"]).mean(),
    }
    return table, summary
