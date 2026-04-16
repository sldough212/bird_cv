import numpy as np


def lookup_segment_idx(segment_index: dict, frame: int) -> int:
    """Return the segment index whose [start, end] range contains ``frame``.

    Args:
        segment_index (dict): Mapping of segment index to ``{"start": int, "end": int}``.
        frame (int): Frame number to look up.

    Returns:
        int: The matching segment index, or the last segment if frame is beyond all ranges.
    """
    for seg_idx, bounds in segment_index.items():
        if bounds["start"] <= frame <= bounds["end"]:
            return int(seg_idx)
    return int(max(segment_index.keys(), key=int))


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate the Intersection over Union (IoU) metric for two binary masks.

    Args:
        pred_mask (np.ndarray): Binary mask representing the predicted segmentation.
            Should contain boolean values or 0/1.
        gt_mask (np.ndarray): Binary mask representing the ground-truth segmentation.
            Should have the same shape as `pred_mask`.

    Returns:
        float: The IoU score between 0.0 and 1.0, where 1.0 indicates perfect overlap
        and 0.0 indicates no overlap.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union
