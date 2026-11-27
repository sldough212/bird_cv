"""Cage masks of an inference run against a ``sam2`` run's masks.

Cages never move within a video, so each video's first predicted keyframe is
compared with every keyframe the ``sam2`` run saved for that video; no
frame alignment is needed. Cages are matched on IoU (Hungarian), and a cage
only one side has scores 0.
"""

from typing import Any

import numpy as np
import polars as pl

from bird_cv.core.layout import InferRun
from bird_cv.segmentation.cage_masks import load_keyframes, match_masks


def compare_masks(pred: list[np.ndarray], truth: list[np.ndarray]) -> dict[str, Any]:
    """Matched IoU of predicted and true cages of one frame.

    Returns:
        ``truth_cages``, ``pred_cages``, ``mean_iou`` and ``min_iou`` over
        every cage of either side (unmatched cages score 0).
    """
    ious = [iou for _, _, iou in match_masks(pred, truth)]
    ious += [0.0] * (max(len(pred), len(truth)) - len(ious))
    return {
        "truth_cages": len(truth),
        "pred_cages": len(pred),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
    }


def evaluate_segments(
    run: InferRun, truth: InferRun, cameras: set[str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Score every video of ``cameras`` ``run`` segmented against ``truth``.

    Args:
        run: The inference run to score.
        truth: The ``sam2`` run whose masks are ground truth.
        cameras: Cameras to score (e.g. the test split).

    Returns:
        One row per (video, truth keyframe), and the summary: ``videos``,
        ``mean_iou``, ``worst_cage_iou`` (mean of per-segment minimums) and
        ``count_accuracy`` (segments with the true cage count).

    Raises:
        FileNotFoundError: If no video was scored, or ``truth`` lacks one.
    """
    rows = []
    for path in sorted((run.dir / "segments").glob("*/*.npz")):
        camera_id, video_id = path.parent.name, path.stem
        if camera_id not in cameras:
            continue
        pred = list(load_keyframes(path)[0][1].values())
        for start, masks in load_keyframes(truth.segments(camera_id, video_id)):
            rows.append(
                {"camera_id": camera_id, "video_id": video_id, "keyframe": start}
                | compare_masks(pred, list(masks.values()))
            )
    if not rows:
        raise FileNotFoundError(
            f"No segmented videos of {sorted(cameras)} in {run.dir}"
        )
    table = pl.DataFrame(rows)
    summary = {
        "videos": table.select("camera_id", "video_id").n_unique(),
        "mean_iou": table["mean_iou"].mean(),
        "worst_cage_iou": table["min_iou"].mean(),
        "count_accuracy": (table["truth_cages"] == table["pred_cages"]).mean(),
    }
    return table, summary
