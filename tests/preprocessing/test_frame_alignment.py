"""Frame alignment of cage and bird-crop media (docs/adr/0002)."""

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import polars as pl

from bird_cv.core.frames import FrameMap, frame_map_path
from bird_cv.preprocessing.annotations_to_yolo import crop_bird, process_item
from bird_cv.preprocessing.crop import crop_cages
from bird_cv.segmentation.frames import iter_canonical_frames


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


def test_crop_cages_keep_empty_frames_keeps_frame_count(tmp_path: Path) -> None:
    """With keep_empty_frames, an empty-mask frame becomes a placeholder."""
    n_frames, size = 3, 32
    _write_video(tmp_path / "videos" / "cam" / "vid.mp4", [128] * n_frames, size)

    guidance_path = tmp_path / "guidance.parquet"
    pl.DataFrame(
        {"video_path": ["cam/vid.mp4"], "target_frames": [[1, 2, 3]]}
    ).write_parquet(guidance_path)

    # Frames 1-2 have a cage mask; frame 3's mask is empty
    seg_dir = tmp_path / "segments" / "cam" / "vid"
    seg_dir.mkdir(parents=True)
    full = np.zeros((size, size), dtype=bool)
    full[8:24, 8:24] = True
    (seg_dir / "segment_index.json").write_text(
        json.dumps({"0": {"start": 1, "end": 2}, "1": {"start": 3, "end": 3}})
    )
    (seg_dir / "0_segmentation.json").write_text(json.dumps({"0": full.tolist()}))
    (seg_dir / "1_segmentation.json").write_text(
        json.dumps({"0": np.zeros_like(full).tolist()})
    )

    def run(out: Path, **kwargs: bool) -> int:
        crop_cages(
            split_guidance_path=guidance_path,
            video_segments_path=tmp_path / "segments",
            clip_output_path=out,
            videos_path=tmp_path / "videos",
            **kwargs,
        )
        return len(list((out / "cam" / "vid" / "0").glob("*.jpg")))

    assert run(tmp_path / "dropped") == n_frames - 1
    assert run(tmp_path / "kept", keep_empty_frames=True) == n_frames


class _FakeCapture:
    """cv2.VideoCapture stand-in returning a few tiny frames at 15 fps."""

    def __init__(self, n_frames: int) -> None:
        self.frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(n_frames)]

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, prop_id: int) -> int:
        return 15

    def release(self) -> None:
        pass


def test_bird_crop_frame_map_matches_boxed_frames(tmp_path: Path) -> None:
    """The frame map lists the canonical frame of each boxed (cropped) frame."""
    videos_dir = tmp_path / "videos"
    (videos_dir / "cam" / "vid").mkdir(parents=True)
    box = {"x": 25, "y": 25, "width": 20, "height": 20}
    item = {
        "data": {"video": "videos/cam/vid/0.mp4"},
        "annotations": [
            {
                "result": [
                    {
                        "value": {
                            "labels": ["bird"],
                            "sequence": [{"frame": f, **box} for f in (2, 3)],
                        }
                    }
                ]
            }
        ],
    }
    guidance_path = tmp_path / "guidance.parquet"
    pl.DataFrame(
        {
            "video_path": ["raw/cam/vid.mp4"],
            "target_frames": [[1, 2, 3, 4]],
            "fps": [15],
            "split": ["train"],
        }
    ).write_parquet(guidance_path)

    with patch(
        "bird_cv.preprocessing.annotations_to_yolo.cv2.VideoCapture",
        return_value=_FakeCapture(4),
    ):
        process_item(
            item=item,
            path_to_videos=videos_dir,
            path_to_output=tmp_path / "crops",
            path_to_guidance=guidance_path,
            crop_size=32,
            frame_map_root=tmp_path / "bbox_videos",
        )

    crops = sorted(
        (tmp_path / "crops" / "images" / "train" / "cam" / "vid" / "0").glob("*.jpg")
    )
    frame_map = FrameMap.read(
        frame_map_path(tmp_path / "bbox_videos" / "cam" / "vid" / "0.mp4")
    )
    # Crop files are named by 1-indexed video frame
    assert [int(p.stem) - 1 for p in crops] == list(frame_map.canonical)
    assert frame_map.canonical == (1, 2)


def test_iter_canonical_frames_numbers_from_zero(tmp_path: Path) -> None:
    """Frames are numbered from 0 and ``frames`` selects canonical frames."""
    video_path = tmp_path / "vid.mp4"
    _write_video(video_path, [0, 120, 240])

    all_frames = list(iter_canonical_frames(video_path))
    selected = list(iter_canonical_frames(video_path, frames=[0, 2]))

    assert [i for i, _ in all_frames] == [0, 1, 2]
    assert [i for i, _ in selected] == [0, 2]
    assert abs(float(selected[1][1].mean()) - 240) < 10
