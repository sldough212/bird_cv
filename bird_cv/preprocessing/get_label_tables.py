"""Module for getting label tables."""

import cv2
import polars as pl
from pathlib import Path


def buffer_frame_data(frame_data, num_frames):
    """Buffer frame data by camera."""
    frame_data = frame_data.with_columns(
        behavior_length=pl.col("frame_end") - pl.col("frame_begin"),
    )

    # Determine buffer size
    frame_data = frame_data.with_columns(
        pl.when(pl.col("behavior_length") < num_frames)
        .then(((num_frames - pl.col("behavior_length")) / 2).cast(pl.Int64))
        .otherwise(pl.lit(0))
        .alias("buffer")
    )

    # Adjust behavior starts and ends
    frame_data = frame_data.with_columns(
        frame_begin=pl.col("frame_begin") - pl.col("buffer"),
        frame_end=pl.col("frame_end") + pl.col("buffer"),
    ).drop("buffer", "behavior_length")

    return frame_data


def get_label_tables(
    label_json_path: Path,
    videos_path: Path,
    output_dir: Path,
    num_frames: int = 16,
) -> None:
    """Process a Label Studio JSON export and generate separate video and frame tables.

    This function reads a Label Studio JSON export, processes the nested annotations,
    and creates two NDJSON files:
    1. `video_data.ndjson` – contains video-level metadata
    2. `frame_data.ndjson` – contains frame-level annotations with bounding boxes and labels

    Args:
        label_json_path (Path): Path to the Label Studio JSON export file.
        output_dir (Path): Directory where the resulting NDJSON files will be saved.
            If the directory does not exist, it will be created along with any
            necessary parent directories.
        num_frames (int): Number of frames used in video classification training.
            Behaviors with less than num_frames will be buffered evenly.

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

    # Grab the true fps from the video
    all_fps = []
    for row in videos.iter_rows(named=True):
        full_video_path = (
            videos_path / row["camera_id"] / f"{row['video_id']}.{row['video_ext']}"
        )
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        all_fps.append(fps)
        cap.release()

    videos = videos.with_columns(
        ls_fps=pl.col("framesCount") / pl.col("duration"), true_fps=pl.Series(all_fps)
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

    # Groupby the track id to get the frame beginning and end
    frames = frames.group_by("track_id").agg(
        pl.col("video_idx").first().alias("video_idx"),
        pl.col("label").first().alias("label"),
        pl.col("frame").min().alias("frame_begin"),
        pl.col("frame").max().alias("frame_end"),
        pl.col("framesCount").first().alias("framesCount"),
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

    # Correct frame numbers
    videos = videos.with_columns(
        framesCount=(
            pl.col("framesCount") * (pl.col("true_fps") / pl.col("ls_fps"))
        ).cast(pl.Int64)
    ).drop("ls_fps")

    frames = frames.with_columns(
        frame_begin=(
            pl.col("frame_begin") * (pl.col("true_fps") / pl.col("ls_fps"))
        ).cast(pl.Int64),
        frame_end=(pl.col("frame_end") * (pl.col("true_fps") / pl.col("ls_fps"))).cast(
            pl.Int64
        ),
    ).drop("ls_fps")

    # Buffer the frame_data behavior if not at least num_frames long
    frames = buffer_frame_data(frame_data=frames, num_frames=num_frames)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    videos.write_ndjson(output_dir / "video_data.ndjson")
    frames.write_ndjson(output_dir / "frame_data.ndjson")
