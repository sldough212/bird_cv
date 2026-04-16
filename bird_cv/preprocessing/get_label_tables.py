"""Module for getting label tables."""

import cv2
import polars as pl
from pathlib import Path


def buffer_frame_data(
    frame_data: pl.DataFrame, video_data: pl.DataFrame, num_frames: int
) -> pl.DataFrame:
    """Extend short behavior clips symmetrically to a minimum length of ``num_frames``.

    For each behavior whose frame span is shorter than ``num_frames``, adds an equal
    buffer to both ``frame_begin`` and ``frame_end``. If the buffer pushes a boundary
    beyond the video extent, the excess is redistributed to the opposite end, and the
    result is clamped to ``[1, framesCount]``.

    Args:
        frame_data: DataFrame containing per-behavior frame annotations. Must include
            columns ``video_id``, ``camera_id``, ``frame_begin``, and ``frame_end``.
        video_data: DataFrame containing video metadata. Must include columns
            ``video_id``, ``camera_id``, and ``framesCount``.
        num_frames: Minimum number of frames each behavior clip should span.

    Returns:
        DataFrame with ``frame_begin`` and ``frame_end`` adjusted so each behavior
        spans at least ``num_frames`` frames, clamped to valid video boundaries.
    """
    frame_data = frame_data.with_columns(
        behavior_length=pl.col("frame_end") - pl.col("frame_begin") + 1,
    )

    # Determine buffer size
    frame_data = frame_data.with_columns(
        pl.when(pl.col("behavior_length") < num_frames)
        .then(((num_frames - pl.col("behavior_length") + 1) // 2).cast(pl.Int64))
        .otherwise(pl.lit(0))
        .alias("buffer")
    )

    # Adjust behavior starts and ends
    frame_data = frame_data.with_columns(
        frame_begin=pl.col("frame_begin") - pl.col("buffer"),
        frame_end=pl.col("frame_end") + pl.col("buffer"),
    ).drop("buffer", "behavior_length")

    # Adjust cases where buffer exceeds video start or finish
    frame_data_with_counts = frame_data.join(
        video_data.select("video_id", "camera_id", "framesCount"),
        on=["video_id", "camera_id"],
        how="left",
    )

    # If frame_begin is negative add to column to add to frame_end
    # If frame_end exceeds the framesCount, subract from frame_begin
    frame_data_with_counts = frame_data_with_counts.with_columns(
        pl.when(pl.col("frame_begin") < 0)
        .then(pl.col("frame_begin").abs() + 1)
        .otherwise(pl.lit(0))
        .alias("add_to_end"),
        pl.when(pl.col("frame_end") > pl.col("framesCount"))
        .then(pl.col("frame_end") - pl.col("framesCount"))
        .otherwise(pl.lit(0))
        .alias("subtract_from_begin"),
    )

    frame_data_with_counts = frame_data_with_counts.with_columns(
        (pl.col("frame_begin") - pl.col("subtract_from_begin"))
        .clip(lower_bound=1)
        .alias("frame_begin"),
        (pl.col("frame_end") + pl.col("add_to_end"))
        .clip(upper_bound=pl.col("framesCount"))
        .alias("frame_end"),
    ).drop("add_to_end", "subtract_from_begin")

    return frame_data_with_counts


def get_label_tables(
    label_json_path: Path,
    videos_path: Path,
    output_dir: Path,
    num_frames: int = 16,
) -> None:
    """Process a Label Studio JSON export and generate separate video and frame tables.

    Reads a Label Studio JSON export, reads true FPS from each video file via cv2,
    FPS-corrects all frame numbers, and buffers short behavior clips to ``num_frames``.
    Writes two NDJSON files:

    1. ``video_data.ndjson`` — video-level metadata including true FPS and frame count
    2. ``frame_data.ndjson`` — per-behavior frame annotations with FPS-corrected and
       buffered ``frame_begin`` / ``frame_end``

    Args:
        label_json_path (Path): Path to the Label Studio JSON export file.
        videos_path (Path): Base directory containing the original video files,
            organized as ``videos_path / camera_id / video_file``. Used to read
            true FPS via cv2.
        output_dir (Path): Directory where the resulting NDJSON files will be saved.
            Created along with any necessary parent directories if it does not exist.
        num_frames (int): Minimum number of frames each behavior clip should span.
            Clips shorter than this are buffered symmetrically. Defaults to 16.

    Returns:
        None: The function writes two files to disk and does not return anything.
    """
    # Process the raw annotations
    annotations = pl.read_json(label_json_path)
    annotations = (
        annotations.filter(pl.col("total_annotations") > 0)
        .rename({"inner_id": "video_idx"})
        .select("data", "video_idx", "annotations")
        .unnest("data")
        .explode("annotations")
        .unnest("annotations")
        .drop("id")
        .explode("result")
        .unnest("result")
        .unnest("value")
        .rename({"video": "video_path"})
    )

    # Subselect video metadata
    videos = (
        annotations.select("video_idx", "video_path", "framesCount", "duration")
        .unique(subset="video_idx")
        .with_columns(
            camera_id=pl.col("video_path")
            .str.split("/")
            .list[-2]
            .str.replace_all("%2C", ","),
            video_id=pl.col("video_path")
            .str.split("/")
            .list[-1]
            .str.splitn(".", 2)
            .struct.field("field_0"),
            video_ext=pl.col("video_path").str.split(".").list[-1],
        )
    )

    # Grab the true fps and frame count from the video
    all_fps = []
    all_frame_counts = []
    for row in videos.iter_rows(named=True):
        full_video_path = (
            videos_path / row["camera_id"] / f"{row['video_id']}.{row['video_ext']}"
        )
        cap = cv2.VideoCapture(full_video_path)
        all_fps.append(cap.get(cv2.CAP_PROP_FPS))
        all_frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    videos = videos.with_columns(
        ls_fps=pl.col("framesCount") / pl.col("duration"),
        true_fps=pl.Series(all_fps),
        true_frame_count=pl.Series(all_frame_counts),
    )

    # Subselect frame metadata
    frames = (
        annotations.select("video_idx", "sequence", "labels", "id", "framesCount")
        .with_columns(labels=pl.col("labels").list[0])
        .explode("sequence")
        .unnest("sequence")
        .drop_nulls(subset="frame")
        .rename({"labels": "label", "id": "track_id"})
        .select(
            "video_idx",
            "track_id",
            "label",
            "frame",
            "x",
            "y",
            "width",
            "height",
            "framesCount",
        )
    )

    # Convert labels to yolo
    frames = frames.with_columns(
        ((pl.col("x") + pl.col("width") / 2) / 100).alias("x"),
        (pl.col("width") / 100).alias("width"),
        ((pl.col("y") + pl.col("height") / 2) / 100).alias("y"),
        (pl.col("height") / 100).alias("height"),
    )

    # Groupby the track id to get the frame beginning and end
    # Also determine the coordinate centroid to be used in assigning cage_ids to tracks
    frames = frames.group_by("track_id").agg(
        pl.col("video_idx").first().alias("video_idx"),
        pl.col("label").first().alias("label"),
        (pl.col("frame").min() + 1).alias("frame_begin"),
        (pl.col("frame").max() + 1).alias("frame_end"),
        pl.col("framesCount").first().alias("framesCount"),
        pl.col("x").mean().alias("mean_x"),
        pl.col("y").mean().alias("mean_y"),
        pl.col("width").mean().alias("mean_width"),
        pl.col("height").mean().alias("mean_height"),
    )
    frames = frames.with_columns(
        pl.when(pl.col("frame_begin") == pl.col("frame_end"))
        .then(pl.col("framesCount"))
        .otherwise(pl.col("frame_end"))
        .alias("frame_end")
    ).drop("framesCount")

    # Join back on video_id to get the camera_id and video_id onto frames
    frames = frames.join(
        videos.select("video_idx", "video_id", "camera_id", "ls_fps", "true_fps"),
        on=["video_idx"],
        how="left",
    ).drop("video_idx")

    videos = videos.drop("video_idx")

    # Use OpenCV frame count directly — avoids rounding errors from duration-based computation
    videos = videos.with_columns(framesCount=pl.col("true_frame_count")).drop(
        "ls_fps", "true_frame_count"
    )

    frames = frames.with_columns(
        frame_begin=(
            pl.col("frame_begin") * (pl.col("true_fps") / pl.col("ls_fps"))
        ).cast(pl.Int64),
        frame_end=(pl.col("frame_end") * (pl.col("true_fps") / pl.col("ls_fps"))).cast(
            pl.Int64
        ),
    ).drop("ls_fps")

    # Truncate resting tracks to num_frames so they contribute one clip each
    frames = frames.with_columns(
        pl.when(pl.col("label") == "Resting")
        .then(pl.col("frame_begin") + num_frames - 1)
        .otherwise(pl.col("frame_end"))
        .alias("frame_end")
    )

    # Buffer the frame_data behavior if not at least num_frames long
    frames = buffer_frame_data(
        frame_data=frames, video_data=videos, num_frames=num_frames
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    videos.write_ndjson(output_dir / "video_data.ndjson")
    frames.write_ndjson(output_dir / "frame_data.ndjson")
