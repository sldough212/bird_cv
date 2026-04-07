import numpy as np


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
