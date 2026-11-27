"""Split guidance generation: assign videos/frames to train, val, and test."""

from pathlib import Path

import cv2
import numpy as np
import polars as pl


def simulate_split_guidance(videos_path: Path, output_path: Path) -> None:
    """Build a split guidance parquet covering every frame of every video found under videos_path.

    Recursively scans for video files, reads their FPS and frame count via OpenCV,
    and writes a parquet with columns ``video_path``, ``fps``, and ``target_frames``
    (a list of all frame indices for each video). Intended as a stand-in for the
    real split guidance produced by the preprocessing pipeline when all frames
    should be processed.

    Args:
        videos_path: Root directory to search recursively for video files.
        output_path: Destination path for the output parquet file.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    video_path_store, fps_store, target_frames_store = [], [], []
    for p in videos_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_extensions:
            # Determine fps of the video
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Find frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            video_path_store.append(str(p))
            fps_store.append(fps)
            target_frames_store.append(list(range(1, frame_count + 1)))

    split_guidance = pl.DataFrame(
        {
            "video_path": video_path_store,
            "fps": fps_store,
            "target_frames": target_frames_store,
        }
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    split_guidance.write_parquet(output_path)


def split_guidance_within_camera(
    path_to_guidance: Path,
    split_ratio: dict[str, float],
    output_path: Path,
) -> None:
    """Randomly assign each video in a split guidance table to train/val/test.

    Unlike :func:`split_camera_data`, assignment is drawn independently per
    video rather than grouped by camera, so a single camera's videos can end
    up split across train, val, and test.

    Args:
        path_to_guidance: Path to a split guidance parquet (e.g. produced by
            :func:`simulate_split_guidance`) containing at least a
            ``video_path`` column.
        split_ratio: Mapping of split name (``"train"``, ``"val"``,
            ``"test"``) to the probability a given video is assigned to that
            split. Values should sum to 1.
        output_path: Destination path for the output parquet, with a
            ``split`` column added.

    Returns:
        None. Writes the updated split guidance parquet to ``output_path``.
    """
    split_guidance = pl.read_parquet(path_to_guidance)

    # Extract a camera_id
    split_guidance = split_guidance.with_columns(
        camera_id=pl.col("video_path")
        .str.split("/")
        .list.get(-2)
        .str.replace_all("%2C", ","),
    )

    splits = np.random.choice(
        list(split_ratio.keys()),
        p=list(split_ratio.values()),
        replace=True,
        size=split_guidance.height,
    )
    split_guidance = split_guidance.with_columns(split=splits)

    split_guidance = split_guidance.drop("camera_id")
    split_guidance.write_parquet(output_path)
