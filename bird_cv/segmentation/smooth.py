import cv2
import numpy as np


def mask_to_filled_polygon(mask, epsilon_ratio=0.01, min_area=250):
    """
    Convert a noisy binary mask into clean filled polygon(s),
    preserving multiple disconnected regions.

    Args:
        mask (np.ndarray): Binary mask (0 or 255)
        epsilon_ratio (float): Controls polygon simplification
        min_area (float): Filter out tiny noise blobs

    Returns:
        np.ndarray: Cleaned boolean mask
    """
    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask.astype(bool)

    clean_mask = np.zeros_like(mask)

    for contour in contours:
        # 🚫 Skip tiny noise
        if cv2.contourArea(contour) < min_area:
            continue

        # Smooth this contour
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Fill it
        cv2.fillPoly(clean_mask, [polygon], 255)

    return clean_mask.astype(bool)


def smooth_masks(
    video_segments: dict,
    epsilon_ratio: float = 0.01,
) -> dict:
    """
    Smooth segmentation masks for a given frame by converting each mask
    into a filled polygon approximation.

    Args:
        video_segments (dict):
            Nested dictionary of segmentation masks structured as:
            {
                frame_idx: {
                    video_id: np.ndarray (mask)
                }
            }
            Each mask is expected to be a 2D or 3D array (e.g., shape
            `(H, W)` or `(1, H, W)`), typically boolean or binary.
        epsilon_ratio (float, optional):
            Approximation factor controlling polygon simplification.
            Passed to `cv2.approxPolyDP` as a fraction of contour perimeter.
            - Lower values → more detailed contours
            - Higher values → smoother, simpler shapes
            Default is 0.01.

    Returns:
        dict:
            A dictionary with the same structure for the specified frame:
            {
                frame_idx: {
                    video_id: np.ndarray (bool mask)
                }
            }
            Each output mask is a smoothed boolean mask representing
            the filled polygon of the original segmentation.
    """
    output_segments: dict[int, dict[int, np.ndarray]] = {}
    for frame, cages in video_segments.items():
        output_segments[frame] = {}
        for video, mask in cages.items():
            output_segments[frame][video] = mask_to_filled_polygon(
                mask=mask.squeeze(), epsilon_ratio=epsilon_ratio
            )

    return output_segments
