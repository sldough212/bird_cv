from collections.abc import Collection, Iterator
from pathlib import Path

import cv2
import numpy as np


def extract_all_frames(
    video_path: Path, output_path: Path, frames: list[int] | None = None
) -> None:
    """Extracts all frames from a video file and saves them as images.

    Args:
        video_path (Path): Path to the input video file to be processed.
        output_path (Path): Directory where extracted frames will be saved.
        frames (list[int] | optional): List of specific frames to only extract.
    """
    output_path.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frames is not None and frame_count not in frames:
            frame_count += 1
            continue

        frame_count += 1
        out_path = output_path / f"{frame_count:05d}.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()


def iter_canonical_frames(
    video_path: Path, frames: Collection[int] | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Read a video's frames in memory, numbered as canonical frames.

    Unlike :func:`extract_all_frames` (whose 1-indexed file names follow
    Label Studio), frames are numbered from 0, as in
    ``docs/adr/0002-canonical-frame-index.md``.

    Args:
        video_path: Path to the (standardized) video.
        frames: Canonical frames to yield; all frames if None.

    Yields:
        ``(canonical_frame, frame)`` pairs in order, ``frame`` in BGR.

    Raises:
        OSError: If the video can't be opened.
    """
    wanted = None if frames is None else set(frames)
    last = None if wanted is None else max(wanted, default=-1)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise OSError(f"Could not open video {video_path}")
        frame_idx = 0
        while last is None or frame_idx <= last:
            success, frame = cap.read()
            if not success:
                break
            if wanted is None or frame_idx in wanted:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def extract_canonical_frames(
    video_path: Path, output_path: Path, frames: Collection[int] | None = None
) -> None:
    """Save a video's frames as JPEGs named by canonical frame (``00000.jpg``, ...).

    Args:
        video_path: Path to the (standardized) video.
        output_path: Directory where frames are saved.
        frames: Canonical frames to save; all frames if None.
    """
    output_path.mkdir(exist_ok=True, parents=True)
    for frame_idx, frame in iter_canonical_frames(video_path, frames):
        cv2.imwrite(str(output_path / f"{frame_idx:05d}.jpg"), frame)
