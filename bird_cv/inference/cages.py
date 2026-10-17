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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bird_cv.core.ids import cage_id_from_index
from bird_cv.preprocessing.crop import crop_cage_frame
from bird_cv.segmentation.segment import find_segment_boundaries, set_sam_and_predict
from bird_cv.segmentation.smooth import smooth_masks

SEGMENT_IOU_THRESHOLD = 0.85
"""Per-cage IoU below which a new mask keyframe starts; ``segment()``'s default."""

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


@dataclass
class CageMasks:
    """Keyframe cage masks of one video, looked up by canonical frame.

    Attributes:
        keyframes: :data:`Keyframe` entries sorted by start frame; the first
            starts at frame 0. Each mask holds from its start frame until
            the next keyframe.
    """

    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def cage_ids(self) -> list[str]:
        """Cage ids in config order."""
        ids: dict[str, None] = {}
        for _, masks in self.keyframes:
            ids.update(dict.fromkeys(masks))
        return sorted(ids, key=int)

    def at(self, frame: int) -> dict[str, np.ndarray]:
        """Masks in effect at canonical ``frame``."""
        starts = [start for start, _ in self.keyframes]
        return self.keyframes[bisect.bisect_right(starts, frame) - 1][1]

    def cage_frame_shapes(self) -> dict[str, tuple[int, int]]:
        """Per cage, the ``(height, width)`` every cage frame is padded to.

        The largest cage crop over all keyframes, rounded up to even sides,
        as :func:`bird_cv.preprocessing.image_utils.images_to_video` pads a
        cage video. Cages whose mask is empty in every keyframe are omitted.
        """
        shapes: dict[str, tuple[int, int]] = {}
        for _, masks in self.keyframes:
            for cage_id, mask in masks.items():
                crop = crop_cage_frame(mask.astype(np.uint8), mask)
                if crop is None:
                    continue
                h, w = crop.shape[:2]
                prev_h, prev_w = shapes.get(cage_id, (0, 0))
                shapes[cage_id] = (max(prev_h, h), max(prev_w, w))
        return {c: (h + h % 2, w + w % 2) for c, (h, w) in shapes.items()}


def cage_frame(
    frame: np.ndarray, mask: np.ndarray, shape: tuple[int, int]
) -> np.ndarray | None:
    """Crop and mask one cage, padded bottom/right with black to ``shape``.

    Returns None for an empty mask: that cage-frame is ``no_bird``.
    """
    crop = crop_cage_frame(frame, mask)
    if crop is None:
        return None
    h, w = crop.shape[:2]
    return cv2.copyMakeBorder(
        crop, 0, shape[0] - h, 0, shape[1] - w, cv2.BORDER_CONSTANT, value=0
    )


class Sam2CageSegmenter:
    """Segments chunks of one camera's standardized video with SAM2.

    Each chunk is segmented independently, re-prompted from the camera's
    reference frame exactly as ``segment()`` does (reference frame as frame
    0, polygon smoothing, then one keyframe per stable segment via
    :func:`find_segment_boundaries`).

    Args:
        predictor: A built SAM2 video predictor, reused across chunks.
        sam_model_id: Hugging Face id of the SAM2 checkpoint (only used if
            ``predictor`` is None).
        config_path: The camera's SAM2 prompt config.
        reference_frame_path: The frame the prompts were drawn on.
        device: Device for SAM2.
        iou_threshold: See :data:`SEGMENT_IOU_THRESHOLD`.
    """

    def __init__(
        self,
        predictor: Any,
        sam_model_id: str,
        config_path: Path,
        reference_frame_path: Path,
        device: str = "cuda",
        iou_threshold: float = SEGMENT_IOU_THRESHOLD,
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
        h, w = frames[0].shape[:2]
        return [(start, {cage_id_from_index(0): np.ones((h, w), dtype=bool)})]
