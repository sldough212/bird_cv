"""Build a behavior clip index by joining frame labels, video metadata, and FPS guidance."""

from pathlib import Path

import polars as pl


def build_clip_index(
    frame_data_path: Path,
    video_data_path: Path,
    split_guidance_path: Path,
    output_path: Path,
) -> None:
    """Join behavior labels with video metadata to produce a behavior clip index.

    Joins ``frame_data_full.ndjson`` (per-track behavior labels) with
    ``video_data_full.ndjson`` (video metadata) and ``split_guidance.parquet``
    (train/val/test split assignment). Frame numbers are already FPS-corrected
    upstream in ``get_label_tables`` so no rescaling is performed here.

    Args:
        frame_data_path: Path to ``frame_data_full.ndjson`` containing columns
            ``track_id``, ``video_id``, ``camera_id``, ``label``, ``frame_begin``,
            ``frame_end``.
        video_data_path: Path to ``video_data_full.ndjson`` containing columns
            ``video_id``, ``camera_id``, ``video_path``.
        split_guidance_path: Path to ``split_guidance.parquet`` containing columns
            ``video_path`` and ``split``.
        output_path: Path where the output parquet will be written.
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
