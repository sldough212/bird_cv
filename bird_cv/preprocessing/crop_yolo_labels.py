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


def _lookup_segment_idx(segment_index: dict, frame: int) -> int:
    """Return the segment index whose [start, end] range contains ``frame``."""
    for seg_idx, bounds in segment_index.items():
        if bounds["start"] <= frame <= bounds["end"]:
            return int(seg_idx)
    # Fall back to the last segment if frame is beyond the index
    return int(max(segment_index.keys(), key=int))


def crop_yolo(
    split: str,
    yolo_data_path: Path,
    yolo_output_path: Path,
    video_segments_path: Path,
    frame: int,
    video_id: str,
    camera_id: str,
) -> list[str]:
    """Crop and save a single frame's image and labels per cage using SAM2 segmentation masks.

    For a given video frame, loads the full-frame YOLO image and labels, looks up
    the appropriate SAM2 segmentation mask via the segment index, and for each cage
    crops the image to the cage region and renormalizes the bounding box labels.
    Saves cropped images and YOLO txt files to ``yolo_output_path``.

    Args:
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``. Controls output
            directory structure (test frames are nested by camera/video/cage).
        yolo_data_path: Root of the full-frame YOLO dataset containing ``images/``
            and ``labels/`` subdirectories.
        yolo_output_path: Root of the per-cage cropped YOLO output dataset.
        video_segments_path: Path to the directory containing SAM2 segmentation JSONs
            organized as ``camera_id / video_id / segment_index.json``.
        frame: Frame number to process.
        video_id: Video identifier.
        camera_id: Camera identifier (decoded, e.g. ``"H7,I22"``).

    Returns:
        List of cage IDs that were successfully processed for this frame.
    """
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
    seg_dir = video_segments_path / camera_id / video_id
    seg_index_path = seg_dir / "segment_index.json"
    if not seg_index_path.exists():
        print(
            f"Skipping frame {frame}: segment_index.json not found at {seg_index_path}"
        )
        return []
    with seg_index_path.open("r") as f:
        segment_index = json.load(f)

    segment_idx = _lookup_segment_idx(segment_index, frame)
    seg_path = seg_dir / f"{segment_idx}_segmentation.json"

    if not seg_path.exists():
        print(f"Skipping frame {frame}: segmentation file not found at {seg_path}")
        return []
    with seg_path.open("r") as f:
        try:
            cage_masks = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping frame {frame}: could not parse {seg_path}")
            return []

    cage_ids_processed = []
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
            / f"{yolo_camera_id}.{video_id}_frame_{frame:05d}_cage_{cage_id}.jpg"
        )

        # Save YOLO txt
        with open(
            label_output_path
            / f"{yolo_camera_id}.{video_id}_frame_{frame:05d}_cage_{cage_id}.txt",
            "w",
        ) as f:
            for label in labels_crop:
                f.write(" ".join(map(str, label)) + "\n")

        cage_ids_processed.append(cage_id)

    return cage_ids_processed


def run_crop_yolo(
    split_guidance_path: Path,
    yolo_data_path: Path,
    yolo_output_path: Path,
    video_segments_path: Path,
    behavior_clips_path: Path | None = None,
    behavior_output_path: Path | None = None,
) -> None:
    """Crop YOLO labels and images per cage, and optionally assign behavior clips to cages.

    Args:
        split_guidance_path: Path to the split guidance.
        yolo_data_path: Root of the full-frame YOLO dataset.
        yolo_output_path: Root of the per-cage cropped YOLO output dataset.
        video_segments_path: Path to the segmentation JSON files.
        behavior_clips_path: Optional path to the parquet output of
            :func:`build_clip_index`. If provided, each processed frame/cage will
            be matched against behavior clips and saved to ``behavior_output_path``.
        behavior_output_path: Path to write the behavior clip index with cage_id
            assigned. Required if ``behavior_clips_path`` is provided.
    """
    behavior_clips = None
    if behavior_clips_path is not None:
        assert behavior_output_path is not None, (
            "behavior_output_path must be provided when behavior_clips_path is set"
        )
        behavior_clips = pl.read_parquet(behavior_clips_path)
        behavior_records: list[dict] = []

    # Load in labeling guidance
    split_guidance = pl.read_parquet(split_guidance_path)
    for row in split_guidance.iter_rows(named=True):
        target_frames = [int(x) for x in sorted(row["target_frames"])]

        camera_id, video_name = extract_camera_video(row["video_path"])
        video_id = Path(video_name).stem
        split = row["split"]

        # Pre-filter behavior clips for this camera/video
        if behavior_clips is not None:
            video_behaviors = behavior_clips.filter(
                (pl.col("camera_id") == camera_id) & (pl.col("video_id") == video_id)
            )

        for frame in target_frames:
            cage_ids = crop_yolo(
                split=split,
                yolo_data_path=yolo_data_path,
                yolo_output_path=yolo_output_path,
                video_segments_path=video_segments_path,
                frame=frame,
                camera_id=camera_id,
                video_id=video_id,
            )

            if behavior_clips is not None and cage_ids:
                for behavior in video_behaviors.iter_rows(named=True):
                    if behavior["frame_begin"] <= frame <= behavior["frame_end"]:
                        for cage_id in cage_ids:
                            behavior_records.append(
                                {
                                    "track_id": behavior["track_id"],
                                    "camera_id": camera_id,
                                    "video_id": video_id,
                                    "cage_id": cage_id,
                                    "label": behavior["label"],
                                    "frame_begin": behavior["frame_begin"],
                                    "frame_end": behavior["frame_end"],
                                    "split": split,
                                }
                            )

    if behavior_clips is not None:
        assert behavior_output_path is not None
        result = pl.DataFrame(behavior_records).unique(subset=["track_id", "cage_id"])
        behavior_output_path.parent.mkdir(exist_ok=True, parents=True)
        result.write_parquet(behavior_output_path)
