"""Per-cage image and YOLO label cropping utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

from bird_cv.preprocessing.image_utils import crop_and_mask_image
from bird_cv.segmentation.frames import extract_all_frames
from bird_cv.segmentation.utils import lookup_segment_idx
from bird_cv.utils import extract_camera_video


def crop_cage_frame(
    frame: np.ndarray | Image.Image,
    cage_mask: list | np.ndarray,
    padding: int = 5,
) -> np.ndarray | None:
    """Crop one in-memory frame to one cage and black out pixels outside it.

    The per-frame step of :func:`crop_cages`, usable on frames held in memory
    (e.g. during inference). Channel order is preserved: an RGB frame gives an
    RGB crop, a BGR frame a BGR crop.

    Args:
        frame: Full frame, as an ``(H, W[, C])`` array or a PIL image.
        cage_mask: 2D boolean mask for the cage, the same size as ``frame``.
        padding: Pixels to keep around the mask's bounding box.

    Returns:
        The cropped, masked frame, or None if the mask is empty (camera
        starting up or cage significantly obstructed).
    """
    mask = np.asarray(cage_mask, dtype=bool)
    if not mask.any():
        return None
    cropped_img, _ = crop_and_mask_image(frame, mask, black_out=True, padding=padding)
    return np.asarray(cropped_img)


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


def crop_cages(
    split_guidance_path: Path,
    video_segments_path: Path,
    clip_output_path: Path,
    videos_path: Path,
) -> None:
    """Crop each cage region from every target frame and save as individual JPEGs.

    For each video in the split guidance, extracts all frames to a temporary
    directory, loads the corresponding SAM2 cage masks, and saves a cropped
    JPEG per cage per frame under ``clip_output_path/{camera_id}/{video_id}/{cage_id}/``.
    Frames with missing segmentation masks, or whose cage mask is empty, are
    skipped.

    Args:
        split_guidance_path: Path to the split guidance parquet produced by
            ``simulate_split_guidance`` or the preprocessing pipeline.
        video_segments_path: Root directory of SAM2 segmentation JSON outputs,
            structured as ``{camera_id}/{video_id}/``.
        clip_output_path: Root directory where cropped cage images will be saved.
        videos_path: Root directory containing the source video files.
    """
    split_guidance = pl.read_parquet(split_guidance_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for video_str, target_frames in split_guidance.select(
            "video_path", "target_frames"
        ).to_numpy():
            camera_id, video_name = extract_camera_video(video_str=video_str)
            video_id = Path(video_name).stem
            image_output_path = clip_output_path / camera_id / video_id
            print(f"  Cropping {camera_id}/{video_id}")

            frame_store_path = temp_path / camera_id / video_id
            frame_store_path.mkdir(exist_ok=True, parents=True)
            extract_all_frames(
                video_path=videos_path / camera_id / video_name,
                output_path=frame_store_path,
            )

            # Extract segmentation info
            seg_dir = video_segments_path / camera_id / video_id
            seg_index_path = seg_dir / "segment_index.json"
            with seg_index_path.open("r") as f:
                segment_index = json.load(f)

            # Crop each cage in camera / video
            for frame in sorted(target_frames):
                frame_cage_masks = _load_cage_masks(seg_dir, segment_index, frame)
                if frame_cage_masks is None:
                    continue

                img = Image.open(frame_store_path / f"{frame:05d}.jpg")

                for cage_id, cage_mask in frame_cage_masks.items():
                    # An empty mask (camera starting up or significantly obstructed)
                    # gives no crop: skip it
                    cropped = crop_cage_frame(img, cage_mask)
                    if cropped is None:
                        continue

                    crop_output = image_output_path / cage_id
                    crop_output.mkdir(exist_ok=True, parents=True)

                    Image.fromarray(cropped).save(crop_output / f"{frame:05d}.jpg")
