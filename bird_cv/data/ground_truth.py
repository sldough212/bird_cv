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
from pathlib import Path
from typing import Any

import cv2
import polars as pl

from bird_cv.client.evaluate_label_studio_tracking import _select_annotation
from bird_cv.core.frames import is_aligned_with_standard, ls_to_canonical
from bird_cv.core.ids import parse_media_path, parse_raw_video_path
from bird_cv.core.labels import NO_BIRD, validate_class
from bird_cv.core.schemas import GROUND_TRUTH_SCHEMA

logger = logging.getLogger(__name__)

# Version marker in prepare's ground-truth skip key, not a parameter default.
GT_VERSION = 1
"""Bump when the ground truth rules change, so prepare rebuilds it."""

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
        overshoot_classify_tasks: Classify tasks overshooting the boxed
            frames by more than ``overshoot_tolerance``, excluded.
        classify_dropped_with_detect: Classify tasks dropped because their
            Detect task was excluded.
        misaligned_detect_tasks: Detect tasks failing
            :func:`~bird_cv.core.frames.is_aligned_with_standard`, excluded.
        unchecked_detect_tasks: Detect tasks whose standardized frame count
            was unknown, so dropped frames could not be checked.
        skipped_detect_tasks: Detect tasks with no boxes and no aligned
            sibling cage to take the frame count from, excluded.
    """

    table: pl.DataFrame
    clipped_frames: int = 0
    conflict_frames: int = 0
    unmatched_classify_tasks: int = 0
    overshoot_classify_tasks: int = 0
    classify_dropped_with_detect: int = 0
    misaligned_detect_tasks: int = 0
    unchecked_detect_tasks: int = 0
    skipped_detect_tasks: int = 0

    def counts(self) -> dict[str, int]:
        """Every count, keyed by attribute name."""
        return {k: v for k, v in vars(self).items() if k != "table"}


@dataclass
class DetectTask:
    """Boxes of one Detect task, keyed by 1-indexed Label Studio frame."""

    boxes: dict[int, Box]
    frames_count: int | None
    duration: float | None


def annotation_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Results of the task's most recent non-cancelled annotation (empty if none).

    Uses the same rule as the tracking evaluation
    (:func:`bird_cv.client.evaluate_label_studio_tracking._select_annotation`).
    """
    annotation = _select_annotation(task.get("annotations", []))
    return annotation.get("result", []) if annotation else []


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

    Raises:
        ValueError: If a sequence skips frames after an enabled keyframe,
            i.e. the export was not interpolated.
    """
    parsed = DetectTask(boxes={}, frames_count=None, duration=None)
    for result in annotation_results(task):
        if result.get("type") != "videorectangle":
            continue
        value = result["value"]
        parsed.frames_count = value["framesCount"]
        parsed.duration = value["duration"]
        sequence = sorted(value["sequence"], key=lambda s: s["frame"])
        for step, following in zip(sequence, sequence[1:]):
            if step.get("enabled", True) and following["frame"] != step["frame"] + 1:
                raise ValueError(
                    f"{task['data']['video']}: sequence jumps from frame "
                    f"{step['frame']} to {following['frame']}; export the Detect "
                    "project with interpolate_frames=True"
                )
        for step in sequence:
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
    """Behaviors on each frame of a Classify task's bird-crop video.

    Args:
        task: One task of a Classify export.

    Returns:
        For each 1-indexed crop frame inside any ``timelinelabels`` range
        (out-of-range frames such as -1 included), the behaviors labeled on
        it, one entry per covering range.

    Raises:
        ValueError: If a label is not a known class.
    """
    labels: dict[int, list[str]] = defaultdict(list)
    for result in annotation_results(task):
        if result.get("type") != "timelinelabels":
            continue
        for label in result["value"]["timelinelabels"]:
            validate_class(label)
            for frame_range in result["value"]["ranges"]:
                for frame in range(frame_range["start"], frame_range["end"] + 1):
                    labels[frame].append(label)
    return labels


def standard_frame_counts(
    raw_videos_dir: Path, extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
) -> dict[tuple[str, str], int] | None:
    """Frame counts of the standardized videos, read from their headers.

    Args:
        raw_videos_dir: Directory of standardized videos laid out as
            ``{camera}/{video}.{ext}``.
        extensions: Lower-case file suffixes counted as videos.

    Returns:
        ``{(camera_id, video_id): frame_count}``, or None if the directory
        doesn't exist.
    """
    if not raw_videos_dir.is_dir():
        return None
    counts = {}
    for path in raw_videos_dir.rglob("*"):
        if path.suffix.lower() in extensions:
            cap = cv2.VideoCapture(str(path))
            counts[parse_raw_video_path(path)] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    return counts


def _by_key(tasks: list[dict[str, Any]], project: str) -> dict[CageKey, dict]:
    """Index tasks by cage key, raising on duplicate media."""
    by_key: dict[CageKey, dict] = {}
    for task in tasks:
        key = parse_media_path(task["data"]["video"])
        if key in by_key:
            raise ValueError(f"Duplicate {project} task for {key}")
        by_key[key] = task
    return by_key


def build_ground_truth(
    detect_tasks: list[dict[str, Any]],
    classify_tasks: list[dict[str, Any]],
    standard_counts: dict[tuple[str, str], int] | None = None,
    overshoot_tolerance: int = 8,
) -> GroundTruthResult:
    """Convert Detect and Classify exports to per-frame ground truth.

    Every canonical frame ``0..framesCount-1`` of every aligned Detect task
    gets one row. A Detect task with no boxes (an empty cage) takes its frame
    count from an aligned cage of the same video and is all ``no_bird``.

    Args:
        detect_tasks: Parsed interpolated Detect export.
        classify_tasks: Parsed Classify export.
        standard_counts: ``{(camera_id, video_id): frame_count}`` of the
            standardized videos, to catch cage videos with dropped frames.
            Tasks whose video is missing (or all tasks, if None) are only
            checked for fps and counted as unchecked.
        overshoot_tolerance: Classify tasks whose ranges run more than this
            many frames past the boxed-frame count are excluded (the crop
            and the Detect boxes no longer correspond); smaller overshoots
            are clipped.

    Returns:
        The ground truth table and counts of clipped, conflicting and
        excluded data.

    Raises:
        ValueError: On duplicate media in either export, or a
            non-interpolated Detect export.
    """
    classify_by_key = _by_key(classify_tasks, "Classify")
    detect = {
        k: parse_detect_task(t) for k, t in _by_key(detect_tasks, "Detect").items()
    }

    result = GroundTruthResult(table=pl.DataFrame(schema=GROUND_TRUTH_SCHEMA))
    result.unmatched_classify_tasks = len(set(classify_by_key) - set(detect))
    if standard_counts is None:
        logger.warning(
            "No standardized frame counts given; skipping the dropped-frame check"
        )

    aligned: dict[CageKey, int] = {}
    for key, task in detect.items():
        if task.frames_count is None or task.duration is None:
            continue
        standard = (standard_counts or {}).get(key[:2])
        if standard is None:
            result.unchecked_detect_tasks += 1
        if is_aligned_with_standard(task.frames_count, task.duration, standard):
            aligned[key] = task.frames_count
        else:
            logger.warning(
                "Excluding %s: %d frames over %.2fs (standardized: %s) is not "
                "aligned with the standardized video",
                key,
                task.frames_count,
                task.duration,
                standard,
            )
            result.misaligned_detect_tasks += 1
    video_frames_count = {key[:2]: count for key, count in aligned.items()}

    tables = []
    for key in sorted(detect):
        task = detect[key]
        if task.frames_count is not None:
            n_frames = aligned.get(key)
        else:
            n_frames = video_frames_count.get(key[:2])
            if n_frames is None:
                logger.warning("Excluding %s: no boxes and no frame count", key)
                result.skipped_detect_tasks += 1
        if n_frames is None:
            result.classify_dropped_with_detect += key in classify_by_key
            continue

        labels: list[str | None] = [NO_BIRD] * n_frames
        boxes: list[Box | None] = [None] * n_frames
        boxed_ls = sorted(task.boxes)
        for ls_frame in boxed_ls:
            labels[ls_to_canonical(ls_frame)] = None
            boxes[ls_to_canonical(ls_frame)] = task.boxes[ls_frame]

        behaviors_by_frame = (
            behavior_by_crop_frame(classify_by_key[key])
            if key in classify_by_key
            else {}
        )
        overshoot = max(behaviors_by_frame, default=0) - len(boxed_ls)
        if overshoot > 0:
            logger.info(
                "%s: Classify ranges overshoot %d boxed frames by %d",
                key,
                len(boxed_ls),
                overshoot,
            )
        if overshoot > overshoot_tolerance:
            logger.warning("Excluding Classify task %s: overshoot %d", key, overshoot)
            result.overshoot_classify_tasks += 1
            behaviors_by_frame = {}
        for crop_frame, behaviors in behaviors_by_frame.items():
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
    logger.info("Ground truth: %d frames; %s", result.table.height, result.counts())
    return result
