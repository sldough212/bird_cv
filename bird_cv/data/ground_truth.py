"""Per-frame ground truth on canonical frames from the Detect and Classify exports.

Chain (see ``docs/adr/0002-canonical-frame-index.md``): Classify frame ``k``
of a bird-crop video is the ``k``-th boxed frame of the matching Detect task
(both 1-indexed Label Studio frames, boxed frames taken from the interpolated
export), which converts to a canonical frame with
:func:`bird_cv.core.frames.ls_to_canonical`. Unboxed canonical frames are
``no_bird``; boxed frames with no behavior are null (a labeling hole).
Detect tasks not aligned frame-for-frame with their standardized video are
excluded and counted.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import polars as pl

from bird_cv.core.frames import is_aligned_with_standard, ls_to_canonical
from bird_cv.core.ids import parse_media_path
from bird_cv.core.labels import NO_BIRD, validate_class
from bird_cv.core.schemas import GROUND_TRUTH_SCHEMA

logger = logging.getLogger(__name__)

CageKey = tuple[str, str, str]
"""``(camera_id, video_id, cage_id)``."""

Box = tuple[float, float, float, float]
"""``(left, top, width, height)`` normalized 0-1 to the cage video."""


@dataclass
class GroundTruthResult:
    """Ground truth plus counts of what the conversion had to fix or drop.

    Attributes:
        table: One row per canonical frame of every included Detect task, in
            :data:`~bird_cv.core.schemas.GROUND_TRUTH_SCHEMA`.
        clipped_frames: Classify frames outside ``[1, n_boxed]`` that were
            dropped (e.g. a range starting at -1 or overshooting the crop).
        conflict_frames: Boxed frames given two different behaviors; set to
            null.
        unmatched_classify_tasks: Classify tasks with no Detect task, skipped.
        misaligned_detect_tasks: Detect tasks failing
            :func:`~bird_cv.core.frames.is_aligned_with_standard`, excluded.
        skipped_detect_tasks: Detect tasks with no boxes and no aligned
            sibling cage to take the frame count from, excluded.
    """

    table: pl.DataFrame
    clipped_frames: int = 0
    conflict_frames: int = 0
    unmatched_classify_tasks: int = 0
    misaligned_detect_tasks: int = 0
    skipped_detect_tasks: int = 0


@dataclass
class DetectTask:
    """Boxes of one Detect task, keyed by 1-indexed Label Studio frame."""

    boxes: dict[int, Box]
    frames_count: int | None
    duration: float | None


def _annotation_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Results of the task's first non-cancelled annotation (empty if none)."""
    for annotation in task.get("annotations", []):
        if not annotation.get("was_cancelled"):
            return annotation.get("result", [])
    return []


def parse_detect_task(task: dict[str, Any]) -> DetectTask:
    """Collect every boxed frame of an interpolated Detect task.

    A frame is boxed if it appears in any ``videorectangle`` region's
    sequence (including a region's final, disabled keyframe). Where regions
    overlap, the first region's box is kept.

    Args:
        task: One task of an interpolated Detect export.

    Returns:
        Boxes by frame, plus ``framesCount`` and ``duration`` (None if the
        task has no regions).
    """
    parsed = DetectTask(boxes={}, frames_count=None, duration=None)
    for result in _annotation_results(task):
        if result.get("type") != "videorectangle":
            continue
        value = result["value"]
        parsed.frames_count = value["framesCount"]
        parsed.duration = value["duration"]
        for step in value["sequence"]:
            if 1 <= step["frame"] <= parsed.frames_count:
                parsed.boxes.setdefault(
                    step["frame"],
                    (
                        step["x"] / 100,
                        step["y"] / 100,
                        step["width"] / 100,
                        step["height"] / 100,
                    ),
                )
    return parsed


def behavior_by_crop_frame(task: dict[str, Any]) -> dict[int, list[str]]:
    """Behaviors on each 1-indexed frame of a Classify task's bird-crop video."""
    labels: dict[int, list[str]] = defaultdict(list)
    for result in _annotation_results(task):
        if result.get("type") != "timelinelabels":
            continue
        for label in result["value"]["timelinelabels"]:
            validate_class(label)
            for frame_range in result["value"]["ranges"]:
                for frame in range(frame_range["start"], frame_range["end"] + 1):
                    labels[frame].append(label)
    return labels


def build_ground_truth(
    detect_tasks: list[dict[str, Any]], classify_tasks: list[dict[str, Any]]
) -> GroundTruthResult:
    """Convert Detect and Classify exports to per-frame ground truth.

    Every canonical frame ``0..framesCount-1`` of every aligned Detect task
    gets one row. A Detect task with no boxes (an empty cage) takes its frame
    count from an aligned cage of the same video and is all ``no_bird``.

    Args:
        detect_tasks: Parsed interpolated Detect export.
        classify_tasks: Parsed Classify export.

    Returns:
        The ground truth table and counts of clipped, conflicting and
        excluded data.
    """
    classify_by_key = {parse_media_path(t["data"]["video"]): t for t in classify_tasks}
    detect = {
        parse_media_path(t["data"]["video"]): parse_detect_task(t) for t in detect_tasks
    }

    result = GroundTruthResult(table=pl.DataFrame(schema=GROUND_TRUTH_SCHEMA))
    result.unmatched_classify_tasks = len(set(classify_by_key) - set(detect))

    aligned: dict[CageKey, int] = {}
    for key, task in detect.items():
        if task.frames_count is None or task.duration is None:
            continue
        if is_aligned_with_standard(task.frames_count, task.duration):
            aligned[key] = task.frames_count
        else:
            logger.warning(
                "Excluding %s: %d frames over %.2fs is not aligned with %s",
                key,
                task.frames_count,
                task.duration,
                "the standardized video",
            )
            result.misaligned_detect_tasks += 1
    video_frames_count = {key[:2]: count for key, count in aligned.items()}

    tables = []
    for key in sorted(detect):
        task = detect[key]
        if task.frames_count is not None:
            if key not in aligned:
                continue
            n_frames = aligned[key]
        elif key[:2] in video_frames_count:
            n_frames = video_frames_count[key[:2]]
        else:
            logger.warning("Excluding %s: no boxes and no frame count", key)
            result.skipped_detect_tasks += 1
            continue

        labels: list[str | None] = [NO_BIRD] * n_frames
        boxes: list[Box | None] = [None] * n_frames
        boxed_ls = sorted(task.boxes)
        for ls_frame in boxed_ls:
            labels[ls_to_canonical(ls_frame)] = None
            boxes[ls_to_canonical(ls_frame)] = task.boxes[ls_frame]

        if key in classify_by_key:
            for crop_frame, behaviors in behavior_by_crop_frame(
                classify_by_key[key]
            ).items():
                if not 1 <= crop_frame <= len(boxed_ls):
                    result.clipped_frames += 1
                    continue
                if len(set(behaviors)) > 1:
                    result.conflict_frames += 1
                    continue
                labels[ls_to_canonical(boxed_ls[crop_frame - 1])] = behaviors[0]

        camera_id, video_id, cage_id = key
        bbox_columns = ("bbox_left", "bbox_top", "bbox_width", "bbox_height")
        tables.append(
            pl.DataFrame(
                {
                    "camera_id": camera_id,
                    "video_id": video_id,
                    "cage_id": cage_id,
                    "frame": range(n_frames),
                    "label": labels,
                    **{
                        name: [b[i] if b else None for b in boxes]
                        for i, name in enumerate(bbox_columns)
                    },
                },
                schema=GROUND_TRUTH_SCHEMA,
            )
        )

    if tables:
        result.table = pl.concat(tables)
    logger.info(
        "Ground truth: %d frames; %d Classify frames clipped to the crop, "
        "%d conflicting frames set to null, %d unmatched Classify tasks, "
        "%d misaligned and %d unsized Detect tasks excluded",
        result.table.height,
        result.clipped_frames,
        result.conflict_frames,
        result.unmatched_classify_tasks,
        result.misaligned_detect_tasks,
        result.skipped_detect_tasks,
    )
    return result
