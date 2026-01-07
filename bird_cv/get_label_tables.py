"""Module for getting label tables."""

import polars as pl
from pathlib import Path


def get_label_tables(
    label_json_path: Path,
    output_dir: Path,
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

    Returns:
        None: The function writes two files to disk and does not return anything.
    """
    # Process the raw annotations
    annotations = pl.read_json(label_json_path)
    annotations = (
        annotations.filter(pl.col("total_annotations") > 0)
        .rename({"inner_id": "video_id"})
        .select("data", "video_id", "annotations")
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
        annotations.select("video_id", "video_path", "framesCount", "duration")
        .unique(subset="video_id")
        .with_columns(camera_id=pl.col("video_path").str.split("/").list[-2])
    )

    # Subselect frame metadata
    frames = (
        annotations.select("video_id", "sequence", "labels", "id", "framesCount")
        .with_columns(labels=pl.col("labels").list[0])
        .explode("sequence")
        .unnest("sequence")
        .drop_nulls(subset="frame")
        .rename({"labels": "label", "id": "track_id"})
        .select(
            "video_id",
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
        pl.col("video_id").first().alias("video_id"),
        pl.col("label").first().alias("label"),
        pl.col("frame").first().alias("frame_begin"),
        pl.col("frame").last().alias("frame_end"),
        pl.col("framesCount").first().alias("framesCount"),
    )
    frames = frames.with_columns(
        pl.when(pl.col("frame_begin") == pl.col("frame_end"))
        .then(pl.col("framesCount"))
        .otherwise(pl.col("frame_end"))
        .alias("frame_end")
    ).drop("framesCount")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    videos.write_ndjson(output_dir / "video_data.ndjson")
    frames.write_ndjson(output_dir / "frame_data.ndjson")
