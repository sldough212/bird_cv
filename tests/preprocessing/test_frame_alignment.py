"""Frame alignment of cage and bird-crop media (docs/adr/0002)."""

from pathlib import Path

import cv2
import numpy as np

from bird_cv.preprocessing.crop import crop_bird
from bird_cv.preprocessing.standardize import iter_canonical_frames


def _write_video(path: Path, values: list[int], size: int = 32) -> None:
    """Write a tiny 15 fps video whose frame i is filled with ``values[i]``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (size, size)
    )
    for value in values:
        writer.write(np.full((size, size, 3), value, dtype=np.uint8))
    writer.release()


def test_crop_bird_pads_with_black_at_frame_edges() -> None:
    """A crop centered on the top-left corner is padded black above and left."""
    frame = np.full((32, 32, 3), 200, dtype=np.uint8)

    crop = crop_bird(frame, 0.0, 0.0, crop_size=16)

    assert crop.shape == (16, 16, 3)
    assert not crop[:8, :].any()
    assert not crop[:, :8].any()
    assert (crop[8:, 8:] == 200).all()


def test_iter_canonical_frames_numbers_from_zero(tmp_path: Path) -> None:
    """Frames are numbered from 0 and ``frames`` selects canonical frames."""
    video_path = tmp_path / "vid.mp4"
    _write_video(video_path, [0, 120, 240])

    all_frames = list(iter_canonical_frames(video_path))
    selected = list(iter_canonical_frames(video_path, frames=[0, 2]))

    assert [i for i, _ in all_frames] == [0, 1, 2]
    assert [i for i, _ in selected] == [0, 2]
    assert abs(float(selected[1][1].mean()) - 240) < 10
