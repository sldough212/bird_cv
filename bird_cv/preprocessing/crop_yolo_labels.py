from PIL import Image
import json
import numpy as np
from pathlib import Path
import polars as pl

from bird_cv.utils import extract_camera_video


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


def normalize_labels_for_crop(labels, crop_coords, image_shape):
    """
    Convert bird bounding boxes to YOLO format relative to a cropped image.

    Args:
        labels (list): List of [category, [x_center, y_center, width, height]] in normalized coords (0-1)
        crop_coords (tuple): (x_min, y_min, x_max, y_max) of crop in original pixels
        image_shape (tuple): (height, width) of original image in pixels

    Returns:
        normalized_labels (list): List of [category, x_center, y_center, width, height] normalized 0-1
    """
    y_min, x_min = crop_coords[1], crop_coords[0]
    y_max, x_max = crop_coords[3], crop_coords[2]
    img_h, img_w = image_shape

    cropped_w = x_max - x_min
    cropped_h = y_max - y_min

    normalized_labels = []
    for label in labels:
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

    return normalized_labels


def crop_yolo(
    split: str,
    yolo_data_path: Path,
    yolo_output_path: Path,
    video_segments_path: Path,
    frame: str,
    segment_frame: str,
    video_id: str,
    camera_id: str,
) -> None:
    # Create new yolo data store
    base_label_output_path = yolo_output_path / "labels" / split
    base_label_output_path.mkdir(exist_ok=True, parents=True)
    base_image_output_path = yolo_output_path / "images" / split
    base_image_output_path.mkdir(exist_ok=True, parents=True)

    # Extract the labels from the txt file
    yolo_camera_id = camera_id.replace(",", "%2C")
    label_path = (
        yolo_data_path
        / "labels"
        / split
        / f"{yolo_camera_id}.{video_id}_frame_{frame:05d}.txt"
    )
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            labels.append((class_id, bbox))

    # Load the image
    frame_path = (
        yolo_data_path
        / "images"
        / split
        / f"{yolo_camera_id}.{video_id}_frame_{frame:05d}.jpg"
    )
    img = Image.open(frame_path)

    # Load in the segmentation
    seg_path = (
        video_segments_path
        / camera_id
        / video_id
        / f"{segment_frame}_segmentation.json"
    )
    with seg_path.open("r") as f:
        cage_masks = json.load(f)

    for cage_id, cage_mask in cage_masks.items():
        # If split is test, move frames into label_output_path / camera / video / camera
        if split == "test":
            label_output_path = (
                base_label_output_path / camera_id / video_id / str(cage_id)
            )
            label_output_path.mkdir(exist_ok=True, parents=True)
            image_output_path = (
                base_image_output_path / camera_id / video_id / str(cage_id)
            )
            image_output_path.mkdir(exist_ok=True, parents=True)
        else:
            label_output_path = base_label_output_path
            image_output_path = base_image_output_path

        # Crop & optionally black out non-cage pixels
        cropped_img, crop_coords = crop_and_mask_image(
            img, cage_mask, black_out=True, padding=5
        )

        # Normalize labels for YOLO
        labels_crop = normalize_labels_for_crop(
            labels, crop_coords, img.size[::-1]
        )  # img.size = (width, height)

        # Save cropped image
        cropped_img.save(
            image_output_path
            / f"{yolo_camera_id}.{video_id}_frame_{frame}_cage_{cage_id}.jpg"
        )

        # Save YOLO txt
        with open(
            label_output_path
            / f"{yolo_camera_id}.{video_id}_frame_{frame}_cage_{cage_id}.txt",
            "w",
        ) as f:
            for label in labels_crop:
                f.write(" ".join(map(str, label)) + "\n")


def run_crop_yolo(
    corrected_targets_path: Path,
    yolo_data_path: Path,
    yolo_output_path: Path,
    video_segments_path: Path,
) -> None:
    # Load in corrected labeling guidance
    corrected_targets = pl.read_parquet(corrected_targets_path)
    for row in corrected_targets.iter_rows(named=True):
        target_frames = [int(x) for x in sorted(row["corrected_target_frames"])]
        seg_target_map = {
            target_frame: ii for ii, target_frame in enumerate(target_frames)
        }

        camera_id, video_name = extract_camera_video(row["video_path"])
        video_id = Path(video_name).stem
        split = row["split"]

        for frame in target_frames:
            segment_frame = seg_target_map[frame]

            crop_yolo(
                split=split,
                yolo_data_path=yolo_data_path,
                yolo_output_path=yolo_output_path,
                video_segments_path=video_segments_path,
                frame=frame,
                segment_frame=segment_frame,
                camera_id=camera_id,
                video_id=video_id,
            )
