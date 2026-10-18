"""Frame rates and frame counts of raw videos versus their Label Studio media.

Ground truth needs each standardized video's frame count, computed from the
raw video's header with :func:`~bird_cv.preprocessing.standardize.expected_standard_frames`
(the rule ``standardize_video`` follows). Label Studio numbers frames at its
own rate unless the labeling config sets one (a legacy export reports 1502
frames over 60.07 s, 25 fps, for a 15 fps video), so the header fps, the
Label Studio fps (``framesCount / duration``) and 15 must all agree before
Label Studio frames can be read as canonical frames.

Cluster diagnostic, one row per raw video and Detect cage::

    python -m bird_cv.data.check_fps bird_cv/pipelines/configs/config.toml \\
        [--detect-export PATH]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.frames import STANDARD_FPS
from bird_cv.core.ids import parse_media_path, parse_raw_video_path
from bird_cv.data.ground_truth import annotation_results
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
    """Header and Label Studio frame rates and counts, per video and cage.

    Args:
        raw_videos_dir: Raw videos as ``{camera_id}/{video_id}.{ext}``; may
            be missing (header columns are then null).
        detect_tasks: An interpolated Detect export.
        max_frames: Frame cap the standardized videos were made with.

    Returns:
        ``camera_id``, ``video_id``, ``cage_id`` (null for a raw video with
        no Detect task), ``header_fps``, ``header_frames``,
        ``expected_frames`` (standardized), ``ls_fps`` and ``ls_frames``
        (null for a task with no boxes, or no task), sorted.
    """
    headers = []
    if raw_videos_dir.is_dir():
        for path in find_raw_videos(raw_videos_dir):
            fps, frames = video_header(path)
            camera_id, video_id = parse_raw_video_path(path)
            headers.append(
                {
                    "camera_id": camera_id,
                    "video_id": video_id,
                    "header_fps": fps,
                    "header_frames": frames,
                    "expected_frames": expected_standard_frames(
                        frames, fps, max_frames=max_frames
                    )
                    if fps > 0
                    else None,
                }
            )
    ls = []
    for task in detect_tasks:
        camera_id, video_id, cage_id = parse_media_path(task["data"]["video"])
        # framesCount/duration of the annotation ground truth reads; works on
        # non-interpolated exports too
        values = [
            r["value"]
            for r in annotation_results(task)
            if r.get("type") == "videorectangle"
        ]
        frames_count = values[0]["framesCount"] if values else None
        duration = values[0]["duration"] if values else None
        ls.append(
            {
                "camera_id": camera_id,
                "video_id": video_id,
                "cage_id": cage_id,
                "ls_fps": frames_count / duration
                if frames_count and duration
                else None,
                "ls_frames": frames_count,
            }
        )
    header_schema = {
        "camera_id": pl.String,
        "video_id": pl.String,
        "header_fps": pl.Float64,
        "header_frames": pl.Int64,
        "expected_frames": pl.Int64,
    }
    ls_schema = {
        "camera_id": pl.String,
        "video_id": pl.String,
        "cage_id": pl.String,
        "ls_fps": pl.Float64,
        "ls_frames": pl.Int64,
    }
    return (
        pl.DataFrame(headers, schema=header_schema)
        .join(
            pl.DataFrame(ls, schema=ls_schema),
            on=["camera_id", "video_id"],
            how="full",
            coalesce=True,
        )
        .select(
            "camera_id",
            "video_id",
            "cage_id",
            "header_fps",
            "header_frames",
            "expected_frames",
            "ls_fps",
            "ls_frames",
        )
        .sort("camera_id", "video_id", "cage_id", nulls_last=True)
    )


def frame_rate_problems(
    table: pl.DataFrame, fps: float = STANDARD_FPS, tolerance: float = 0.01
) -> pl.DataFrame:
    """Rows of :func:`frame_rate_table` whose header or Label Studio fps isn't ``fps``.

    Args:
        table: Output of :func:`frame_rate_table`.
        fps: The standardized frame rate.
        tolerance: Allowed relative deviation, as in
            :func:`~bird_cv.core.frames.is_aligned_with_standard`.

    Returns:
        The offending rows (null values are not checked).
    """

    def off(col: str) -> pl.Expr:
        return ((pl.col(col) - fps).abs() > tolerance * fps).fill_null(False)

    return table.filter(off("header_fps") | off("ls_fps"))


def check_frame_rates(table: pl.DataFrame) -> None:
    """Raise, listing every offending video, unless all frame rates are 15.

    Raises:
        ValueError: If :func:`frame_rate_problems` finds any row.
    """
    problems = frame_rate_problems(table)
    if problems.height:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
            listing = str(problems)
        raise ValueError(
            f"{problems.height} video/cage rows have a header or Label Studio fps "
            f"other than {STANDARD_FPS}; Label Studio frames would not be "
            f"canonical frames. Run `python -m bird_cv.data.check_fps CONFIG`.\n"
            f"{listing}"
        )


def standard_counts(table: pl.DataFrame) -> dict[tuple[str, str], int] | None:
    """``{(camera_id, video_id): expected standardized frames}``, None if no headers.

    The input of ``build_ground_truth(standard_counts=...)``.
    """
    rows = (
        table.filter(pl.col("expected_frames").is_not_null())
        .select("camera_id", "video_id", "expected_frames")
        .unique()
    )
    if not rows.height:
        return None
    return {(c, v): n for c, v, n in rows.iter_rows()}


def main() -> None:
    from bird_cv.core.layout import Layout, snapshot_files
    from bird_cv.pipelines.config import load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument(
        "--detect-export",
        type=Path,
        default=None,
        help="Detect export (default: the snapshot of --prepared or the last prepare)",
    )
    parser.add_argument("--prepared", default=None, help="Prepared-data id")
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    export = args.detect_export
    if export is None:
        layout = Layout(cfg.path_to_base_output)
        prepared = layout.resolve_prepared(args.prepared)
        export = snapshot_files(layout.snapshot(prepared.snapshot_hash))[0]
    table = frame_rate_table(
        cfg.path_to_raw_videos,
        json.loads(export.read_text()),
        max_frames=cfg.prepare.standardized_max_frames,
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)
        problems = frame_rate_problems(table)
        mismatched = table.filter(
            pl.col("ls_frames").ne_missing(pl.col("expected_frames"))
            & pl.col("ls_frames").is_not_null()
            & pl.col("expected_frames").is_not_null()
        )
        print(
            f"{table.height} rows; {problems.height} with fps != {STANDARD_FPS}; "
            f"{mismatched.height} cages whose Label Studio framesCount differs "
            "from the expected standardized count (excluded from ground truth)"
        )
        if problems.height:
            print(problems)
    sys.exit(1 if problems.height else 0)


if __name__ == "__main__":
    main()
