from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest

import bird_cv.inference.chunks as chunks
from bird_cv.classification.types import FramePrediction
from bird_cv.core.labels import NO_BIRD
from bird_cv.detection.types import Detection
from bird_cv.inference.cages import CageMasks, cage_frame, check_sam2_configs
from bird_cv.inference.chunks import iter_chunks, stitch_overlapping
from bird_cv.inference.classify import classify_runs
from bird_cv.inference.gap_fill import cage_predictions, fill_gaps
from bird_cv.preprocessing.crop import crop_cage_frame


class EchoClassifier:
    """Labels each input frame with the canonical frame its crop encodes."""

    input_type: ClassVar[Literal["bbox_crop", "cage"]] = "bbox_crop"
    num_frames = 16

    def predict(self, video_path: Path) -> list[FramePrediction]:
        raise NotImplementedError

    def predict_frames(self, frames: Sequence[np.ndarray]) -> list[FramePrediction]:
        assert len(frames) >= self.num_frames
        return [
            FramePrediction(i, f"f{int(c[0, 0])}", 1.0) for i, c in enumerate(frames)
        ]


def _crop(frame: int) -> np.ndarray:
    return np.full((2, 2), frame, dtype=np.int64)


def test_labels_map_back_to_their_canonical_frames() -> None:
    # Long run 0-19, short run 30-34 (padded from neighbors), and 50-52
    # where the whole pool (other frames excluded) is shorter than a window
    frames = list(range(0, 20)) + list(range(30, 35))
    preds = classify_runs(frames, [_crop(f) for f in frames], EchoClassifier())
    assert {f: p.label for f, p in preds.items()} == {f: f"f{f}" for f in frames}

    tiny = [50, 51, 52]
    preds = classify_runs(tiny, [_crop(f) for f in tiny], EchoClassifier())
    assert {f: p.label for f, p in preds.items()} == {f: f"f{f}" for f in tiny}


def test_run_crossing_a_chunk_boundary_stitches_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 40 frames, chunks of 20 overlapping by 4; boxed run 5-34 crosses 20
    monkeypatch.setattr(
        chunks,
        "iter_canonical_frames",
        lambda path: ((i, _crop(i)) for i in range(40)),
    )
    boxed = set(range(5, 35))
    windows = []
    for start, window in iter_chunks(Path("unused"), 20, overlap=4):
        frames = [start + i for i in range(len(window)) if start + i in boxed]
        crops = [window[f - start] for f in frames]
        windows.append(
            (start, start + len(window), classify_runs(frames, crops, EchoClassifier()))
        )
    stitched = stitch_overlapping(windows)
    assert {f: p.label for f, p in stitched.items()} == {f: f"f{f}" for f in boxed}


def test_iter_chunks_prepends_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
        return ((i, _crop(i)) for i in range(25))

    monkeypatch.setattr(chunks, "iter_canonical_frames", fake_frames)
    got = [
        (start, [int(f[0, 0]) for f in frames])
        for start, frames in iter_chunks(Path("unused"), 10, overlap=3)
    ]
    assert got == [
        (0, list(range(0, 10))),
        (7, list(range(7, 20))),
        (17, list(range(17, 25))),
    ]


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


def test_cage_masks_crop_box_matches_full_frame_crop() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    middle, edge, empty = (np.zeros((40, 50), dtype=bool) for _ in range(3))
    middle[10:20, 15:31] = True
    middle[25, 22] = True
    edge[0:6, 44:50] = True  # padding clipped at the frame border
    masks = CageMasks()
    masks.add([(0, {"0": middle, "1": edge, "2": empty})])
    shapes = masks.cage_frame_shapes()

    for cage_id, mask in (("0", middle), ("1", edge)):
        expected = crop_cage_frame(frame, mask)
        got = cage_frame(frame, masks.at(0, cage_id), shapes[cage_id])
        assert expected is not None and got is not None
        h, w = expected.shape[:2]
        assert np.array_equal(got[:h, :w], expected)
        assert not got[h:].any() and not got[:, w:].any()
    assert masks.at(0, "2") is None
