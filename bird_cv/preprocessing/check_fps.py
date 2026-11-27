"""Raw-header and Label Studio frame rates and counts of every Detect task.

Reported by ``bird_cv/pipelines/check_fps.py``.
"""

from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.frames import is_aligned_with_standard
from bird_cv.core.ids import parse_media_path, parse_raw_video_path
from bird_cv.preprocessing.ground_truth import annotation_results
from bird_cv.preprocessing.standardize import (
    expected_standard_frames,
    find_raw_videos,
    video_header,
)


def frame_rate_table(
    raw_videos_dir: Path,
    detect_tasks: list[dict[str, Any]],
    max_frames: int | None = None,
) -> pl.DataFrame:
    """One row per Detect task: raw header, expected standardized and Label Studio frames.

    Args:
        raw_videos_dir: Raw videos as ``{camera_id}/{video_id}.{ext}``.
        detect_tasks: A Detect export.
        max_frames: Frame cap of the standardized videos.

    Returns:
        ``camera_id``, ``video_id``, ``cage_id``, ``header_fps``,
        ``header_frames``, ``expected_frames``, ``ls_fps``, ``ls_frames`` and
        ``aligned`` (null where unknown; videos without a readable header
        have null header columns).
    """
    paths = (
        {parse_raw_video_path(p): p for p in find_raw_videos(raw_videos_dir)}
        if raw_videos_dir.is_dir()
        else {}
    )
    headers: dict[tuple[str, str], tuple[float, int]] = {}
    rows = []
    for task in detect_tasks:
        camera_id, video_id, cage_id = parse_media_path(task["data"]["video"])
        key = (camera_id, video_id)
        if key in paths and key not in headers:
            headers[key] = video_header(paths[key])
        fps, frames = headers.get(key, (0.0, 0))
        expected = (
            expected_standard_frames(frames, fps, max_frames=max_frames)
            if fps > 0
            else None
        )
        value: dict[str, Any] = next(
            (
                r["value"]
                for r in annotation_results(task)
                if r.get("type") == "videorectangle"
            ),
            {},
        )
        ls_frames, duration = value.get("framesCount"), value.get("duration")
        rows.append(
            {
                "camera_id": camera_id,
                "video_id": video_id,
                "cage_id": cage_id,
                "header_fps": fps if fps > 0 else None,
                "header_frames": frames if fps > 0 else None,
                "expected_frames": expected,
                "ls_fps": ls_frames / duration if ls_frames and duration else None,
                "ls_frames": ls_frames,
                "aligned": is_aligned_with_standard(ls_frames, duration, expected)
                if ls_frames and duration
                else None,
            }
        )
    schema = {
        "camera_id": pl.String,
        "video_id": pl.String,
        "cage_id": pl.String,
        "header_fps": pl.Float64,
        "header_frames": pl.Int64,
        "expected_frames": pl.Int64,
        "ls_fps": pl.Float64,
        "ls_frames": pl.Int64,
        "aligned": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema).sort("camera_id", "video_id", "cage_id")


def standard_counts(table: pl.DataFrame) -> dict[tuple[str, str], int] | None:
    """``{(camera_id, video_id): expected frames}`` for ``build_ground_truth``, None if empty.

    Args:
        table: Output of :func:`frame_rate_table`.

    Returns:
        Expected standardized frame counts of videos with a readable header.
    """
    rows = table.drop_nulls("expected_frames").select(
        "camera_id", "video_id", "expected_frames"
    )
    return {(c, v): n for c, v, n in rows.unique().iter_rows()} or None
