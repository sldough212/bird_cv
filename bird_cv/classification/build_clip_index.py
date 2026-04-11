"""Build a behavior clip index by joining frame labels, video metadata, and FPS guidance."""

from pathlib import Path

import polars as pl


def build_clip_index(
    frame_data_path: Path,
    video_data_path: Path,
    split_guidance_path: Path,
    output_path: Path,
) -> None:
    """Join behavior labels with video metadata and FPS-correct frame boundaries.

    Joins ``frame_data_full.ndjson`` (per-track behavior labels) with
    ``video_data_full.ndjson`` (video metadata including camera_id) and
    ``split_guidance.parquet`` (actual video FPS and train/val/test split).
    Frame boundaries from Label Studio are rescaled from Label Studio FPS to
    the actual video FPS.

    Args:
        frame_data_path: Path to ``frame_data_full.ndjson`` containing columns
            ``track_id``, ``video_id``, ``label``, ``frame_begin``, ``frame_end``.
        video_data_path: Path to ``video_data_full.ndjson`` containing columns
            ``video_id``, ``video_path``, ``camera_id``, ``duration``.
        split_guidance_path: Path to ``corrected_frame_guidance`` parquet containing
            columns ``video_path``, ``fps`` (actual video FPS), ``ls_fps`` (Label
            Studio annotation FPS), and ``split``.
        output_path: Path to output parquet
    """
    frame_data = pl.read_ndjson(frame_data_path).select(
        "track_id",
        "video_id",
        "camera_id",
        "label",
        "frame_begin",
        "frame_end",
    )
    video_data = pl.read_ndjson(video_data_path).select(
        "video_id",
        "camera_id",
        "video_path",
    )
    split_guidance = (
        pl.scan_parquet(split_guidance_path).select("video_path", "split").collect()
    )

    joined = frame_data.join(
        video_data, on=["video_id", "camera_id"], how="inner"
    ).join(split_guidance, on="video_path", how="inner")

    joined = joined.select(
        "track_id",
        "camera_id",
        "video_id",
        "label",
        "frame_begin",
        "frame_end",
        "split",
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    joined.write_parquet(output_path)
