from pathlib import Path

import pytest

from bird_cv.classification.types import FramePrediction
from bird_cv.core.labels import NO_BIRD
from bird_cv.detection.types import Detection
from bird_cv.inference.cages import check_sam2_configs
from bird_cv.inference.chunks import stitch_overlapping
from bird_cv.inference.gap_fill import cage_predictions, fill_gaps


def _det(frame: int, conf: float = 0.9) -> Detection:
    return Detection(frame, 1, 0.1 * frame, 0.2, 0.3, 0.4, conf, "bird")


def test_gap_fill_carries_short_gaps_and_marks_the_rest_no_bird() -> None:
    # Gap of 2 (frames 1-2) is filled; gap of 4 (frames 4-7) is not; the
    # low-confidence box at 9 is dropped; frame 11 has an empty cage mask.
    dets = [_det(0), _det(3), _det(8), _det(9, conf=0.1), _det(10), _det(12)]
    boxes = fill_gaps(dets, 14, det_conf=0.25, max_gap=2, empty_frames={11})

    assert boxes[1] == boxes[2] == boxes[0]
    assert all(boxes[f] is None for f in (4, 5, 6, 7, 11, 13))
    assert boxes[9] == boxes[8]  # 9 is now a 1-frame gap

    labels = {f: FramePrediction(f, "resting", 1.0) for f, b in enumerate(boxes) if b}
    df = cage_predictions("H7,I22", "54", "0", boxes, labels)
    assert df["frame"].to_list() == list(range(14))
    no_bird = df.filter(df["label"] == NO_BIRD)
    assert no_bird["frame"].to_list() == [4, 5, 6, 7, 11, 13]
    assert no_bird["bbox_left"].null_count() == no_bird.height
    assert no_bird["confidence"].null_count() == no_bird.height
    # A gap-filled frame carries the source detection's box and det_conf
    assert df.row(2, named=True)["bbox_left"] == boxes[0].left  # type: ignore[union-attr]
    assert df.row(2, named=True)["det_conf"] == 0.9


def test_stitch_splits_overlap_at_midpoint() -> None:
    # Chunks of 10 with an overlap of 4: windows [0,10), [6,20), [16,25)
    windows = [
        (0, 10, {f: "a" for f in range(0, 10)}),
        (6, 20, {f: "b" for f in range(6, 20)}),
        (16, 25, {f: "c" for f in range(16, 25)}),
    ]
    stitched = stitch_overlapping(windows)

    assert sorted(stitched) == list(range(25))
    assert [stitched[f] for f in range(25)] == (["a"] * 8 + ["b"] * 10 + ["c"] * 7)


def test_missing_sam2_config_fails_fast(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "H7,I22.json").write_text("{}")
    (tmp_path / "frames" / "H7,I22" / "54").mkdir(parents=True)
    (tmp_path / "frames" / "H7,I22" / "54" / "00001.jpg").write_bytes(b"")

    check_sam2_configs(tmp_path, ["H7,I22"])
    with pytest.raises(FileNotFoundError, match="H9, I33"):
        check_sam2_configs(tmp_path, ["H7,I22", "I33", "H9"])
