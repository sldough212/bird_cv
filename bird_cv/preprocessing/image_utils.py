"""Image cropping and label normalization utilities for YOLO preprocessing."""

from PIL import Image
import numpy as np


def crop_and_mask_image(img, mask, black_out=True, padding=0):
    """
    Crop an image to the bounding box of the True region in a mask.

    Args:
        img (PIL.Image.Image): Input image (RGB or grayscale)
        mask (np.ndarray): Boolean array same size as image (True = object)
        black_out (bool): Whether to zero out pixels outside mask
        padding (int): Optional pixels to pad around the bounding box

    Returns:
        cropped_img (PIL.Image.Image): Cropped (and masked) image
        crop_coords (tuple): (x_min, y_min, x_max, y_max) in original image coords
    """
    img_np = np.array(img)
    mask = np.array(mask, dtype=bool)

    ys, xs = np.where(mask)
    y_min, y_max = (
        max(ys.min() - padding, 0),
        min(ys.max() + padding, mask.shape[0] - 1),
    )
    x_min, x_max = (
        max(xs.min() - padding, 0),
        min(xs.max() + padding, mask.shape[1] - 1),
    )

    if black_out:
        masked_img = img_np.copy()
        masked_img[~mask] = 0
    else:
        masked_img = img_np

    cropped = masked_img[y_min : y_max + 1, x_min : x_max + 1]
    cropped_img = Image.fromarray(cropped)

    return cropped_img, (x_min, y_min, x_max, y_max)


def normalize_labels_for_crop(
    labels, crop_coords, image_shape, return_winning_idx: bool = False
):
    """
    Convert bird bounding boxes to YOLO format relative to a cropped image.

    Args:
        labels (list): List of [category, [x_center, y_center, width, height]] in normalized coords (0-1)
        crop_coords (tuple): (x_min, y_min, x_max, y_max) of crop in original pixels
        image_shape (tuple): (height, width) of original image in pixels
        return_winning_idx (bool): If True, also return the index of the last label
            whose centroid falls within the crop.

    Returns:
        normalized_labels (list): List of [category, x_center, y_center, width, height] normalized 0-1
    """
    y_min, x_min = crop_coords[1], crop_coords[0]
    y_max, x_max = crop_coords[3], crop_coords[2]
    img_h, img_w = image_shape

    cropped_w = x_max - x_min
    cropped_h = y_max - y_min

    normalized_labels = []
    winning_idx: int | None = None
    for ii, label in enumerate(labels):
        category, bbox = label
        # bbox assumed to be normalized to original image: [x_center, y_center, w, h]
        x0, y0, w, h = bbox

        # Convert to pixel coords in full image
        cx_pixel = x0 * img_w
        cy_pixel = y0 * img_h
        w_pixel = w * img_w
        h_pixel = h * img_h

        # Offset relative to cropped image
        cx_crop = cx_pixel - x_min
        cy_crop = cy_pixel - y_min

        # Normalize relative to cropped image
        cx_norm = cx_crop / cropped_w
        cy_norm = cy_crop / cropped_h
        w_norm = w_pixel / cropped_w
        h_norm = h_pixel / cropped_h

        # Only include labels that are at least partially inside crop
        if 0 <= cx_norm <= 1 and 0 <= cy_norm <= 1:
            normalized_labels.append([category, cx_norm, cy_norm, w_norm, h_norm])
            winning_idx = ii

    if return_winning_idx:
        return normalized_labels, winning_idx
    return normalized_labels
