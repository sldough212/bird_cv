"""Per-cage YOLO image and label cropping pipeline."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from bird_cv.preprocessing.image_utils import (
    crop_and_mask_image,
    normalize_labels_for_crop,
)
from bird_cv.segmentation.utils import lookup_segment_idx


def _load_cage_masks(seg_dir: Path, segment_index: dict, frame: int) -> dict | None:
    """Load cage masks for a given frame from the segmentation JSON.

    Args:
        seg_dir: Directory containing segmentation JSON files.
        segment_index: Mapping of segment index to ``{"start": int, "end": int}``.
        frame: Frame number to look up.

    Returns:
        Mapping of cage_id to 2D boolean mask, or None if the file is missing or invalid.
    """
    seg_path = seg_dir / f"{lookup_segment_idx(segment_index, frame)}_segmentation.json"
    if not seg_path.exists():
        return None
    with seg_path.open("r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def assign_cage(
    mean_x: float,
    mean_y: float,
    cage_masks: dict,
) -> str | None:
    """Determine which cage a track belongs to via centroid pixel lookup.

    Checks whether the track's mean bbox centroid falls within each cage's
    boolean mask. Returns the first matching cage_id.

    Args:
        mean_x: Mean x-center of the track bbox, normalized 0–1.
        mean_y: Mean y-center of the track bbox, normalized 0–1.
        cage_masks: Mapping of cage_id to 2D boolean mask array.

    Returns:
        Matching cage_id string, or None if the centroid is not inside any cage.
    """
    for cage_id, cage_mask in cage_masks.items():
        mask_arr = np.array(cage_mask, dtype=bool)
        img_h, img_w = mask_arr.shape
        cy = int(np.clip(mean_y * img_h, 0, img_h - 1))
        cx = int(np.clip(mean_x * img_w, 0, img_w - 1))
        if mask_arr[cy, cx]:
            return cage_id
    return None


def crop_yolo_frame(
    split: str,
    yolo_data_path: Path,
    yolo_output_path: Path,
    frame: int,
    video_id: str,
    camera_id: str,
    cage_id: str,
    cage_mask: list,
) -> None:
    """Crop and save one frame's image and YOLO labels for a single cage.

    Args:
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
        yolo_data_path: Root of the full-frame YOLO dataset.
        yolo_output_path: Root of the per-cage cropped YOLO output dataset.
        frame: Frame number to process.
        video_id: Video identifier.
        camera_id: Camera identifier (decoded, e.g. ``"H7,I22"``).
        cage_id: Cage identifier.
        cage_mask: 2D boolean mask for this cage.
    """
    yolo_camera_id = camera_id.replace(",", "%2C")

    if split == "test":
        label_output_path = (
            yolo_output_path / "labels" / split / camera_id / video_id / str(cage_id)
        )
        image_output_path = (
            yolo_output_path / "images" / split / camera_id / video_id / str(cage_id)
        )
    else:
        label_output_path = yolo_output_path / "labels" / split
        image_output_path = yolo_output_path / "images" / split
    label_output_path.mkdir(exist_ok=True, parents=True)
    image_output_path.mkdir(exist_ok=True, parents=True)

    stem = f"{yolo_camera_id}.{video_id}_frame_{frame:05d}_cage_{cage_id}"
    label_path = (
        yolo_data_path
        / "labels"
        / split
        / f"{yolo_camera_id}.{video_id}_frame_{frame:05d}.txt"
    )
    image_path = (
        yolo_data_path
        / "images"
        / split
        / f"{yolo_camera_id}.{video_id}_frame_{frame:05d}.jpg"
    )

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            labels.append((int(parts[0]), list(map(float, parts[1:]))))

    img = Image.open(image_path)
    cropped_img, crop_coords = crop_and_mask_image(
        img, cage_mask, black_out=True, padding=5
    )
    labels_crop = normalize_labels_for_crop(labels, crop_coords, img.size[::-1])

    cropped_img.save(image_output_path / f"{stem}.jpg")
    with open(label_output_path / f"{stem}.txt", "w") as f:
        for label in labels_crop:
            f.write(" ".join(map(str, label)) + "\n")


def run_crop_yolo(
    clip_index_path: Path,
    yolo_data_path: Path,
    yolo_output_path: Path,
    video_segments_path: Path,
    clip_output_path: Path | None = None,
) -> None:
    """Crop YOLO labels and images per cage, driven by the behavior clip index.

    For each track in the clip index, assigns the track to a cage using its mean
    bbox centroid, then crops every frame in the clip range to that cage region.
    Crops are deduplicated so frames shared across tracks are only processed once.
    Optionally writes the clip index with ``cage_id`` added.

    Args:
        clip_index_path: Path to the clip index parquet from ``build_clip_index``.
        yolo_data_path: Root of the full-frame YOLO dataset.
        yolo_output_path: Root of the per-cage cropped YOLO output dataset.
        video_segments_path: Path to the SAM2 segmentation JSON files.
        clip_output_path: Optional path to write the clip index with ``cage_id`` added.
    """
    clip_index = pl.read_parquet(clip_index_path)

    # Step 1: Assign cage_id to each track via centroid pixel lookup
    records = []
    for (camera_id, video_id), group in clip_index.group_by(["camera_id", "video_id"]):
        seg_dir = video_segments_path / camera_id / video_id
        seg_index_path = seg_dir / "segment_index.json"
        if not seg_index_path.exists():
            print(f"Skipping {camera_id}/{video_id}: segment_index.json not found")
            continue
        with seg_index_path.open("r") as f:
            segment_index = json.load(f)

        ref_frame = group["frame_begin"].min()
        cage_masks = _load_cage_masks(seg_dir, segment_index, ref_frame)
        if cage_masks is None:
            print(
                f"Skipping {camera_id}/{video_id}: could not load masks at frame {ref_frame}"
            )
            continue

        for track in group.iter_rows(named=True):
            matched_cage_id = assign_cage(track["mean_x"], track["mean_y"], cage_masks)
            records.append({**track, "cage_id": matched_cage_id})

    clip_with_cages = pl.DataFrame(records).filter(pl.col("cage_id").is_not_null())

    if clip_output_path is not None:
        clip_output_path.parent.mkdir(exist_ok=True, parents=True)
        clip_with_cages.write_parquet(clip_output_path)

    # Step 2: Explode to per-frame rows and deduplicate crops
    frames_to_crop = (
        clip_with_cages.with_columns(
            pl.int_ranges(pl.col("frame_begin"), pl.col("frame_end") + 1).alias("frame")
        )
        .explode("frame")
        .select("camera_id", "video_id", "cage_id", "frame", "split")
        .unique(subset=["camera_id", "video_id", "cage_id", "frame"])
    )

    # Step 3: Crop frames grouped by video; cache seg file across consecutive frames
    for (camera_id, video_id), group in frames_to_crop.group_by(
        ["camera_id", "video_id"]
    ):
        seg_dir = video_segments_path / camera_id / video_id
        with (seg_dir / "segment_index.json").open("r") as f:
            segment_index = json.load(f)

        current_seg_idx: int | None = None
        frame_cage_masks: dict | None = None

        for row in group.sort("frame").iter_rows(named=True):
            frame = row["frame"]
            cage_id: str = row["cage_id"]
            split = row["split"]
            yolo_camera_id = camera_id.replace(",", "%2C")

            stem = f"{yolo_camera_id}.{video_id}_frame_{frame:05d}_cage_{cage_id}"
            if split == "test":
                out_img = (
                    yolo_output_path
                    / "images"
                    / split
                    / camera_id
                    / video_id
                    / str(cage_id)
                    / f"{stem}.jpg"
                )
            else:
                out_img = yolo_output_path / "images" / split / f"{stem}.jpg"
            if out_img.exists():
                continue

            seg_idx = lookup_segment_idx(segment_index, frame)
            if seg_idx != current_seg_idx:
                frame_cage_masks = _load_cage_masks(seg_dir, segment_index, frame)
                current_seg_idx = seg_idx

            if frame_cage_masks is None or cage_id not in frame_cage_masks:
                print(f"Skipping frame {frame} cage {cage_id}: mask not available")
                continue

            crop_yolo_frame(
                split=split,
                yolo_data_path=yolo_data_path,
                yolo_output_path=yolo_output_path,
                frame=frame,
                video_id=video_id,
                camera_id=camera_id,
                cage_id=cage_id,
                cage_mask=frame_cage_masks[cage_id],
            )
