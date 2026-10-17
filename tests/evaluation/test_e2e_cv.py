import polars as pl
import pytest

from bird_cv.core.schemas import GROUND_TRUTH_SCHEMA, PREDICTION_SCHEMA
from bird_cv.evaluation.cv import evaluate_cv, fold_split
from bird_cv.evaluation.e2e import evaluate_e2e

BOX = (0.1, 0.1, 0.3, 0.3)


def _truth(rows: list[tuple[str, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(cam, "v", "0", f, label, *BOX) for cam, f, label in rows],
        schema=GROUND_TRUTH_SCHEMA,
        orient="row",
    )


def _pred(rows: list[tuple[str, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(cam, "v", "0", f, label, 0.9, *BOX, 0.8) for cam, f, label in rows],
        schema=PREDICTION_SCHEMA,
        orient="row",
    )


def test_evaluate_e2e_missing_prediction_and_duplicates() -> None:
    truth = _truth([("A", 0, "resting"), ("A", 1, "resting")])
    # Frame 1 has no prediction row: a miss of resting, counted separately.
    result = evaluate_e2e(truth, _pred([("A", 0, "resting")]))

    assert result.n_frames_without_prediction == 1
    assert result.frame.n_scored == 2
    assert result.frame.macro_f1 == pytest.approx(2 / 3)  # TP1 FN1
    assert result.summary()["classes_scored"] == ["resting"]

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_e2e(truth, _pred([("A", 0, "resting"), ("A", 0, "preening")]))


def test_fold_split_is_disjoint() -> None:
    cv_folds = pl.DataFrame({"camera_id": list("ABCDEF"), "fold": [0, 0, 1, 1, 2, 2]})
    for fold in range(3):
        split = fold_split(cv_folds, fold)
        groups = {
            name: set(split.filter(pl.col("split") == name)["camera_id"])
            for name in ("train", "val", "test")
        }
        assert groups["test"] == set(
            cv_folds.filter(pl.col("fold") == fold)["camera_id"]
        )
        assert len(groups["val"]) == 1
        assert not (groups["train"] & groups["val"])
        assert not ((groups["train"] | groups["val"]) & groups["test"])
        assert groups["train"] | groups["val"] | groups["test"] == set("ABCDEF")


def test_evaluate_cv_drops_predictions_on_non_test_cameras() -> None:
    cv_folds = pl.DataFrame({"camera_id": ["A", "B"], "fold": [0, 1]})
    truth = _truth([("A", 0, "resting"), ("B", 0, "resting")])
    right = _pred([("A", 0, "resting"), ("B", 0, "resting")])
    # Fold 0's model is wrong on B, a camera it trained on: must be ignored.
    fold0 = _pred([("A", 0, "resting"), ("B", 0, "preening")])

    cv = evaluate_cv(truth, cv_folds, {0: fold0, 1: right})

    assert cv.pooled.frame.macro_f1 == 1.0
    assert cv.pooled.frame.n_scored == 2
    assert cv.per_fold["n_scored"].to_list() == [1, 1]
