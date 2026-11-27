"""Trained cage segmenter (ADR 0007): labels, static masks, ordering, scoring, flow."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import polars as pl
import pytest
import torch

from bird_cv.core.layout import InferRun
from bird_cv.evaluation.segments import compare_masks, evaluate_segments
from bird_cv.preprocessing.sampling import select_cage_frames
from bird_cv.segmentation.yolo_cages import YoloCageSegmenter
from bird_cv.segmentation.cage_masks import (
    row_major_order,
    save_keyframes,
    static_cage_masks,
    yolo_seg_label,
)
from bird_cv.training.registry import YoloSegTrainer

H, W = 40, 60


def _box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


# A 2x2 grid of cages, top row slightly staggered
GRID = [
    _box(2, 2, 25, 18),
    _box(32, 4, 58, 19),
    _box(2, 22, 25, 38),
    _box(32, 22, 58, 38),
]


def _masks_run(run: InferRun, camera: str, video: str, keyframes: list[list]) -> None:
    """A sam2 run's saved masks for one video: keyframes at 0, 10, 20, ..."""
    save_keyframes(
        run.segments(camera, video),
        [
            (10 * i, {str(c): m for c, m in enumerate(k)})
            for i, k in enumerate(keyframes)
        ],
    )


def _video(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    for i in range(n):
        writer.write(np.full((H, W, 3), i * 8 % 256, dtype=np.uint8))
    writer.release()
    return path


def test_polygon_labels_are_normalized_cage_outlines() -> None:
    lines = yolo_seg_label([GRID[0], np.zeros((H, W), dtype=bool)]).splitlines()
    assert len(lines) == 1  # empty masks get no line
    cls, *coords = lines[0].split()
    xy = np.array(coords, dtype=float).reshape(-1, 2)
    assert cls == "0" and len(xy) >= 4
    assert np.allclose(xy.min(0), [2 / W, 2 / H]) and np.allclose(
        xy.max(0), [24 / W, 17 / H]
    )


def test_static_masks_vote_match_and_order_row_major() -> None:
    assert row_major_order([GRID[3], GRID[0], GRID[2], GRID[1]]) == [1, 3, 2, 0]
    jitter = np.zeros((H, W), dtype=bool)
    jitter[2:18, 2:27] = True  # cage 0 two px wider on one frame of three
    frames = [
        [(GRID[3], 0.9), (GRID[0], 0.8), (GRID[1], 0.9), (GRID[2], 0.3)],
        [(jitter, 0.9), (GRID[1], 0.9), (GRID[3], 0.9)],  # cage 2 missed
        [(GRID[1], 0.9), (GRID[0], 0.9), (GRID[3], 0.9), (GRID[2], 0.4)],
    ]
    masks = static_cage_masks(frames)
    assert all((m == g).all() for m, g in zip(masks, GRID, strict=True))
    # expected_cages keeps the most confident, still row-major; too few raises
    kept = static_cage_masks(frames, expected_cages=3)
    assert all((m == g).all() for m, g in zip(kept, [GRID[0], GRID[1], GRID[3]]))
    with pytest.raises(ValueError, match="Found 4 cages, expected 5"):
        static_cage_masks(frames, expected_cages=5)
    with pytest.raises(ValueError, match="Found 0"):
        static_cage_masks([[], []])


def test_segment_scores_match_cages_and_count_misses(tmp_path: Path) -> None:
    assert compare_masks(GRID[::-1], GRID)["min_iou"] == 1.0
    missed = compare_masks(GRID[:3], GRID)
    assert missed["min_iou"] == 0.0 and missed["mean_iou"] == 0.75

    truth, run = InferRun(tmp_path / "sam2"), InferRun(tmp_path / "yolo")
    _masks_run(truth, "c1", "v1", [GRID, GRID[:3]])
    _masks_run(run, "c1", "v1", [GRID])
    table, summary = evaluate_segments(run, truth, {"c1"})
    assert table.height == 2 and summary["count_accuracy"] == 0.5
    with pytest.raises(FileNotFoundError):
        evaluate_segments(run, truth, {"other"})


def test_cage_frames_flow_from_sam2_masks_to_yolo_data(tmp_path: Path) -> None:
    masks = InferRun(tmp_path / "sam2")
    _masks_run(masks, "c1", "v1", [GRID, GRID[:3]])
    _video(masks.standardized("c1", "v1"), 20)
    raw = tmp_path / "raw" / "c1" / "v1.mp4"  # only its path is read
    other = tmp_path / "raw" / "c9" / "v1.mp4"  # camera not in split
    split = pl.DataFrame({"camera_id": ["c1"], "split": ["train"]})

    frames = select_cage_frames([raw, other], split, masks, per_video=3)
    # Keyframes [0, 9] and [10, 19]: two frames from the first, one from the second
    assert frames["frame"].to_list() == [3, 6, 14]
    assert frames["mask_start"].to_list() == [0, 0, 10]
    with pytest.raises(FileNotFoundError, match="c1/v2"):
        select_cage_frames([tmp_path / "raw/c1/v2.mp4"], split, masks, 3)

    out = tmp_path / "dataset"
    YoloSegTrainer().build_dataset(
        frames, None, None, None, out, tmp_path / "g.parquet"
    )
    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert [p.stem for p in labels] == ["c1_v1_000003", "c1_v1_000006", "c1_v1_000014"]
    # Each frame is labeled with its own keyframe's cages
    assert [len(p.read_text().splitlines()) for p in labels] == [4, 4, 3]
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 3


def test_segmenter_detects_on_sampled_frames_only(tmp_path: Path) -> None:
    calls = []

    def predict(frames, **kwargs):
        calls.append(len(frames))
        masks = torch.from_numpy(np.stack(GRID[::-1])).float()
        conf = torch.full((4,), 0.9)
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=masks), boxes=SimpleNamespace(conf=conf)
            )
        ] * len(frames)

    segmenter = YoloCageSegmenter(SimpleNamespace(predict=predict), sample_frames=5)
    [(start, masks)] = segmenter.segment_video(_video(tmp_path / "c/v.mp4", 30), 12)
    assert calls == [5] and start == 0
    assert list(masks) == ["0", "1", "2", "3"]
    assert all((masks[str(i)] == g).all() for i, g in enumerate(GRID))
