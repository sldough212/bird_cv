"""Cage and bird crops: per-cage frames, bird-centered crops and clip windows."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from PIL import Image

from bird_cv.preprocessing.image_utils import crop_and_mask_image
from bird_cv.segmentation.frames import extract_all_frames
from bird_cv.segmentation.utils import lookup_segment_idx
from bird_cv.core.ids import parse_raw_video_path


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
            camera_id, video_id = parse_raw_video_path(video_str)
            video_name = Path(video_str).name
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


def crop_bird(
    frame: np.ndarray, cx_norm: float, cy_norm: float, crop_size: int = 224
) -> np.ndarray:
    """Crop a fixed square centered on a bird, padding with black past the edges.

    Args:
        frame: Frame as an ``(H, W[, C])`` array (e.g. BGR from cv2).
        cx_norm: Crop center x, normalized 0-1 to the frame width.
        cy_norm: Crop center y, normalized 0-1 to the frame height.
        crop_size: Side length in pixels of the square crop.

    Returns:
        The ``crop_size`` x ``crop_size`` crop (for even ``crop_size``).
    """
    frame_h, frame_w = frame.shape[:2]
    half = crop_size // 2

    cx = int(cx_norm * frame_w)
    cy = int(cy_norm * frame_h)

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - frame_w)
    pad_bottom = max(0, y2 - frame_h)

    crop = frame[max(0, y1) : min(frame_h, y2), max(0, x1) : min(frame_w, x2)]

    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    return crop


def smooth_centers(
    cx: np.ndarray, cy: np.ndarray, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth box-center sequences with an edge-padded moving average.

    The window shrinks to ``min(window, len(cx))`` for short sequences.

    Args:
        cx: Box x-centers, one per frame, in frame order.
        cy: Box y-centers, one per frame, the same length as ``cx``.
        window: Moving-average window length in frames; at least 1.

    Returns:
        Smoothed ``(cx, cy)``, each the same length as the input (empty
        arrays for empty input).

    Raises:
        ValueError: If ``window`` is less than 1.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    n = len(cx)
    if n == 0:
        return np.empty(0), np.empty(0)
    k = min(window, n)
    kernel = np.ones(k) / k
    pad = k // 2
    cx_smooth = np.convolve(np.pad(cx, pad, mode="edge"), kernel, mode="valid")
    cy_smooth = np.convolve(np.pad(cy, pad, mode="edge"), kernel, mode="valid")
    # An even window yields one extra trailing value; drop it
    return cx_smooth[:n], cy_smooth[:n]


def center_window(
    start: int, end: int, num_frames: int, total_frames: int
) -> tuple[int, int]:
    """Symmetrically expand a ``[start, end)`` range to length ``num_frames``.

    Extends into real neighboring frames from the source video on both
    sides so the labeled segment ends up centered in the window, rather
    than pinned to one edge. If one side runs out of room against a video
    boundary, the remaining deficit is shifted to the other side. If the
    video itself has fewer than ``num_frames`` frames, the returned range
    simply covers the whole video (the caller pads the rest).

    Args:
        start: Inclusive start index (0-based) of the labeled segment.
        end: Exclusive end index (0-based) of the labeled segment.
        num_frames: Desired window length.
        total_frames: Number of frames available in the source video.

    Returns:
        ``(new_start, new_end)``, clamped to ``[0, total_frames)``.
    """
    deficit = num_frames - (end - start)
    left_pad = deficit // 2
    right_pad = deficit - left_pad

    new_start = start - left_pad
    new_end = end + right_pad

    if new_start < 0:
        new_end += -new_start
        new_start = 0
    if new_end > total_frames:
        new_start -= new_end - total_frames
        new_end = total_frames

    return max(new_start, 0), new_end
