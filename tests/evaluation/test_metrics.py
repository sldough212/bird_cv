import math

import polars as pl
import pytest

from bird_cv.evaluation.metrics import (
    detection_presence_metrics,
    frame_metrics,
    mot_metrics,
    per_camera_frame_metrics,
)


def _joined() -> pl.DataFrame:
    # Camera A: one resting frame mislabeled head_move.
    # Camera B: a false positive (no_bird -> resting), a miss
    # (head_move -> no_bird) and a labeling hole (null truth).
    return pl.DataFrame(
        {
            "camera_id": ["A", "A", "A", "B", "B", "B"],
            "label": ["resting", "resting", "head_move", "no_bird", "head_move", None],
            "pred_label": [
                "resting",
                "head_move",
                "head_move",
                "resting",
                "no_bird",
                "resting",
            ],
        }
    )


def test_frame_macro_f1_hand_computed() -> None:
    m = frame_metrics(_joined())

    # resting: TP1 FP1 FN1 -> 0.5; head_move: TP1 FP1 FN1 -> 0.5;
    # no_bird: TP0 FP1 FN1 -> 0; other classes absent -> excluded.
    assert m.macro_f1 == pytest.approx(1 / 3)
    assert (m.n_scored, m.n_excluded) == (5, 1)
    f1 = dict(m.per_class.select("label", "f1").rows())
    assert f1["resting"] == pytest.approx(0.5)
    assert f1["no_bird"] == 0.0
    assert f1["preening"] is None
    assert m.confusion.sum() == 5
    nb, rest = m.classes.index("no_bird"), m.classes.index("resting")
    assert m.confusion[nb, rest] == 1

    per_camera = per_camera_frame_metrics(_joined())
    assert per_camera["camera_id"].to_list() == ["A", "B"]
    assert per_camera["macro_f1"].to_list() == pytest.approx([2 / 3, 0.0])
    assert per_camera["n_excluded"].to_list() == [0, 1]


def _boxes(rows: list[tuple | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "camera_id": ["A"] * len(rows),
            "video_id": ["v"] * len(rows),
            "cage_id": ["0"] * len(rows),
            "frame": list(range(len(rows))),
            "bbox_left": [r[0] if r else None for r in rows],
            "bbox_top": [r[1] if r else None for r in rows],
            "bbox_width": [r[2] if r else None for r in rows],
            "bbox_height": [r[3] if r else None for r in rows],
        },
        schema_overrides={c: pl.Float64 for c in ("bbox_left", "bbox_top")},
    )


def test_detection_presence_hand_computed() -> None:
    box = (0.0, 0.0, 0.5, 0.5)
    truth = _boxes([box, box, None, box, None, (0.0, 0.0, 0.4, 0.5)])
    # exact hit; IoU 1/3 (miss + FP); false positive; miss; empty; IoU 0.8
    pred = _boxes([box, (0.25, 0.0, 0.5, 0.5), box, None, None, box])

    m = detection_presence_metrics(truth, pred, iou_threshold=0.5)

    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 2, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["mean_iou"] == pytest.approx(0.9)

    mot = mot_metrics(truth, pred).filter(pl.col("camera_id") == "all")
    # 4 true boxes, 2 misses, 2 false positives, no switches.
    assert mot["mota"].item() == pytest.approx(1 - (2 + 2) / 4)
    assert not math.isnan(mot["idf1"].item())
