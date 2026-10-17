"""Cage segmentation for inference: SAM2 prompt configs, keyframe masks, cage frames.

A camera's SAM2 prompt config lives at
``{segmentation_configs}/configs/{camera_id}.json`` with its reference frame
under ``{segmentation_configs}/frames/{camera_id}/<video>/00001.jpg`` (the
layout :func:`bird_cv.segmentation.segment.run_segment` reads). The config's
keys are bird ids in cage order; a cage is identified by its position
(:func:`bird_cv.core.ids.cage_id_from_index`), never by bird id.
"""

import bisect
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor

from bird_cv.core.ids import cage_id_from_index
from bird_cv.preprocessing.crop import crop_cage_frame
from bird_cv.preprocessing.image_utils import crop_and_mask_image
from bird_cv.segmentation.segment import find_segment_boundaries, set_sam_and_predict
from bird_cv.segmentation.smooth import smooth_masks

Keyframe = tuple[int, dict[str, np.ndarray]]
"""``(first canonical frame, {cage_id: 2D bool mask})`` of one stable mask segment."""


def sam2_config_path(segmentation_configs_path: Path, camera_id: str) -> Path:
    """Path of a camera's SAM2 prompt config."""
    return segmentation_configs_path / "configs" / f"{camera_id}.json"


def sam2_reference_frame(segmentation_configs_path: Path, camera_id: str) -> Path:
    """Path of the frame a camera's SAM2 prompts were drawn on.

    Mirrors ``run_segment``: the first video folder under
    ``frames/{camera_id}/``, frame ``00001.jpg``.

    Raises:
        FileNotFoundError: If the camera has no reference frame folder.
    """
    frames_dir = segmentation_configs_path / "frames" / camera_id
    video_dirs = sorted(p for p in frames_dir.glob("*/") if p.is_dir())
    if not video_dirs:
        raise FileNotFoundError(f"No SAM2 reference frame folder in {frames_dir}")
    return video_dirs[0] / "00001.jpg"


def check_sam2_configs(
    segmentation_configs_path: Path, camera_ids: Iterable[str]
) -> None:
    """Fail fast if any camera lacks a SAM2 prompt config or its reference frame.

    Args:
        segmentation_configs_path: Root holding ``configs/`` and ``frames/``.
        camera_ids: Cameras about to be processed.

    Raises:
        FileNotFoundError: Listing every camera missing a config JSON or
            reference frame, if any.
    """

    def has_config(camera_id: str) -> bool:
        if not sam2_config_path(segmentation_configs_path, camera_id).is_file():
            return False
        try:
            return sam2_reference_frame(segmentation_configs_path, camera_id).is_file()
        except FileNotFoundError:
            return False

    missing = sorted({c for c in camera_ids if not has_config(c)})
    if missing:
        raise FileNotFoundError(
            f"No SAM2 prompt config under {segmentation_configs_path / 'configs'} "
            f"for cameras: {', '.join(missing)}"
        )


@dataclass(frozen=True)
class CageCrop:
    """One cage's crop region in one keyframe.

    Attributes:
        box: ``(x_min, y_min, x_max, y_max)``, inclusive, in full-frame
            pixels: the mask's bounding box plus ``crop_cage_frame``'s
            default padding (5 px).
        mask: The cage mask cut to ``box``.
    """

    box: tuple[int, int, int, int]
    mask: np.ndarray

    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 5) -> "CageCrop | None":
        """Crop region of a full-frame mask, or None if the mask is empty.

        ``padding`` must match :func:`crop_cage_frame`'s, as in :func:`cage_frame`.
        """
        if not mask.any():
            return None
        _, (x0, y0, x1, y1) = crop_and_mask_image(
            mask, mask, black_out=False, padding=padding
        )
        box = (int(x0), int(y0), int(x1), int(y1))
        return cls(box=box, mask=mask[y0 : y1 + 1, x0 : x1 + 1])

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the crop."""
        return self.mask.shape[0], self.mask.shape[1]


class CageMasks:
    """Keyframe cage crops of one video, looked up by canonical frame.

    Only each keyframe's crop box and cropped mask are kept, not the
    full-frame masks, so memory stays small however long the video is.
    """

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.crops: list[dict[str, CageCrop | None]] = []

    def add(self, keyframes: Iterable[Keyframe]) -> None:
        """Append keyframes, in frame order, after those already added."""
        for start, masks in keyframes:
            if self.starts and start <= self.starts[-1]:
                raise ValueError("keyframes must be added in frame order")
            self.starts.append(start)
            self.crops.append({c: CageCrop.from_mask(m) for c, m in masks.items()})

    @property
    def cage_ids(self) -> list[str]:
        """Cage ids in config order."""
        ids: set[str] = set()
        for crops in self.crops:
            ids.update(crops)
        return sorted(ids, key=int)

    def at(self, frame: int, cage_id: str) -> CageCrop | None:
        """Crop of ``cage_id`` in effect at canonical ``frame`` (None if empty)."""
        return self.crops[bisect.bisect_right(self.starts, frame) - 1].get(cage_id)

    def cage_frame_shapes(self) -> dict[str, tuple[int, int]]:
        """Per cage, the ``(height, width)`` every cage frame is padded to.

        The largest cage crop over all keyframes, rounded up to even sides,
        as :func:`bird_cv.preprocessing.image_utils.images_to_video` pads a
        cage video. Cages whose mask is empty in every keyframe are omitted.
        """
        shapes: dict[str, tuple[int, int]] = {}
        for crops in self.crops:
            for cage_id, crop in crops.items():
                if crop is None:
                    continue
                h, w = crop.shape
                prev_h, prev_w = shapes.get(cage_id, (0, 0))
                shapes[cage_id] = (max(prev_h, h), max(prev_w, w))
        return {c: (h + h % 2, w + w % 2) for c, (h, w) in shapes.items()}


def cage_frame(
    frame: np.ndarray, crop: CageCrop | None, shape: tuple[int, int]
) -> np.ndarray | None:
    """Crop and mask one cage, padded bottom/right with black to ``shape``.

    Same pixels as :func:`crop_cage_frame` on the full-frame mask: cutting
    the frame to ``crop.box`` first leaves the mask's bounding box and its
    padding unchanged.

    Returns None for an empty mask: that cage-frame is ``no_bird``.
    """
    if crop is None:
        return None
    x0, y0, x1, y1 = crop.box
    cropped = crop_cage_frame(frame[y0 : y1 + 1, x0 : x1 + 1], crop.mask)
    if cropped is None:
        return None
    h, w = cropped.shape[:2]
    return cv2.copyMakeBorder(
        cropped, 0, shape[0] - h, 0, shape[1] - w, cv2.BORDER_CONSTANT, value=0
    )


class Sam2CageSegmenter:
    """Segments chunks of one camera's standardized video with SAM2.

    Each chunk is segmented independently, re-prompted from the camera's
    reference frame exactly as ``segment()`` does (reference frame as frame
    0, polygon smoothing, then one keyframe per stable segment via
    :func:`find_segment_boundaries`).

    Args:
        predictor: A built SAM2 video predictor, reused across chunks; if
            None, ``set_sam_and_predict`` builds one per chunk.
        sam_model_id: Hugging Face id of the SAM2 checkpoint (only used if
            ``predictor`` is None).
        config_path: The camera's SAM2 prompt config.
        reference_frame_path: The frame the prompts were drawn on.
        device: Device for SAM2.
        iou_threshold: Per-cage IoU below which a new mask keyframe starts;
            0.85 as in ``segment()``.
    """

    def __init__(
        self,
        predictor: SAM2VideoPredictor | None,
        sam_model_id: str,
        config_path: Path,
        reference_frame_path: Path,
        device: str = "cuda",
        iou_threshold: float = 0.85,
    ) -> None:
        self.predictor = predictor
        self.sam_model_id = sam_model_id
        self.config_path = config_path
        self.reference_frame_path = reference_frame_path
        self.device = device
        self.iou_threshold = iou_threshold

    def segment_chunk(self, frames: Sequence[np.ndarray], start: int) -> list[Keyframe]:
        """Return the keyframes of ``frames`` (BGR), the chunk starting at ``start``."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            shutil.copy(self.reference_frame_path, temp_path / "00000.jpg")
            for i, frame in enumerate(frames, start=1):
                cv2.imwrite(str(temp_path / f"{i:05d}.jpg"), frame)
            video_segments = set_sam_and_predict(
                sam_model_id=self.sam_model_id,
                data_path=temp_path,
                config_path=self.config_path,
                device=self.device,
                predictor=self.predictor,
            )

        video_segments = smooth_masks(video_segments=video_segments)
        # Drop the reference frame and renumber from 0, as segment() does
        video_segments.pop(0)
        video_segments = {ii: v for ii, v in enumerate(video_segments.values())}
        boundaries = find_segment_boundaries(
            video_segments=video_segments, iou_threshold=self.iou_threshold
        )
        return [
            (
                start + b,
                {
                    cage_id_from_index(int(obj_id)): np.asarray(mask, dtype=bool)
                    for obj_id, mask in video_segments[b].items()
                },
            )
            for b in boundaries
        ]


class WholeFrameSegmenter:
    """Treats the whole frame as one cage (``cage_id`` 0); skips SAM2.

    For videos that already show a single cage, e.g. the smoke test.
    """

    def segment_chunk(self, frames: Sequence[np.ndarray], start: int) -> list[Keyframe]:
        """One keyframe at ``start`` whose single mask covers the whole frame.

        Args:
            frames: BGR frames of the chunk; only the first's size is used.
            start: Canonical frame of ``frames[0]``.

        Returns:
            ``[(start, {"0": all-True mask})]``.
        """
        h, w = frames[0].shape[:2]
        return [(start, {cage_id_from_index(0): np.ones((h, w), dtype=bool)})]
