import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from bird_cv.core.labels import BEHAVIORS, NO_BIRD
from bird_cv.core.schemas import GROUND_TRUTH_SCHEMA
from bird_cv.data.ground_truth import build_ground_truth, parse_detect_task
from bird_cv.data.sampling import select_videomae_clips, select_yolo_frames
from bird_cv.data.snapshot import annotation_hash
from bird_cv.data.splits import (
    camera_behaviors,
    choose_initial_split,
    coverage_gaps,
    cv_folds_path,
    split_path,
    update_split,
)
from bird_cv.data.summary import summarize
from bird_cv.pipelines.config import API_KEY_ENV_VAR, load_config
from bird_cv.pipelines.prepare import run_prepare

CONFIG = Path(__file__).parents[2] / "bird_cv/pipelines/configs/config.toml"

URL = "/data/local-files/?d=gscratch/x/{kind}/{camera}/{video}/{cage}.mp4"


def detect_task(
    camera: str, video: str, cage: str, regions: list[tuple[int, int]], n: int
) -> dict[str, Any]:
    """Interpolated Detect task with one region per (first, last) LS frame."""
    result = [
        {
            "id": f"r{i}",
            "type": "videorectangle",
            "value": {
                "framesCount": n,
                "duration": n / 15,
                "sequence": [
                    {"frame": f, "x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
                    for f in range(first, last + 1)
                ],
            },
        }
        for i, (first, last) in enumerate(regions)
    ]
    return {
        "data": {
            "video": URL.format(
                kind="cage_videos", camera=camera, video=video, cage=cage
            )
        },
        "annotations": [
            {
                "id": 1,
                "result": result,
                "created_at": "2026-10-01",
                "updated_at": "2026-10-01",
            }
        ],
        "updated_at": "2026-10-01",
    }


def classify_task(
    camera: str, video: str, cage: str, ranges: list[tuple[str, int, int]]
) -> dict[str, Any]:
    result = [
        {
            "id": f"c{i}",
            "type": "timelinelabels",
            "value": {"ranges": [{"start": s, "end": e}], "timelinelabels": [label]},
        }
        for i, (label, s, e) in enumerate(ranges)
    ]
    return {
        "data": {
            "video": URL.format(
                kind="cropped_bbox_videos", camera=camera, video=video, cage=cage
            )
        },
        "annotations": [{"id": 2, "result": result, "created_at": "2026-10-01"}],
    }


def gt_rows(
    camera: str, video: str, labels: Sequence[str | None], split: str
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "camera_id": camera,
            "video_id": video,
            "cage_id": "0",
            "frame": range(len(labels)),
            "label": labels,
            **{c: None for c in ("bbox_left", "bbox_top", "bbox_width", "bbox_height")},
            "split": split,
        },
        schema={**GROUND_TRUTH_SCHEMA, "split": pl.String},
    )


def test_ground_truth_chain_with_mid_video_gap() -> None:
    # L25/IMG_0016: boxes on LS frames 46-191 and 220-314 (146 + 95 boxed).
    detect = [
        detect_task("L25", "IMG_0016", "0", [(46, 191), (220, 314)], n=314),
        detect_task("L25", "IMG_0016", "1", [], n=0),  # empty cage
    ]
    misaligned = detect_task("L25", "IMG_0099", "0", [(1, 10)], n=60)
    misaligned["annotations"][0]["result"][0]["value"]["duration"] = 2.0  # 30 fps
    classify = [
        classify_task(
            "L25",
            "IMG_0016",
            "0",
            [
                ("resting", -1, 3),
                ("preening", 3, 3),
                ("walking___hopping", 114, 165),
                ("head_move", 240, 245),
            ],
        )
    ]
    result = build_ground_truth(detect + [misaligned], classify)
    gt = result.table.filter(pl.col("cage_id") == "0")
    label = dict(gt.select("frame", "label").iter_rows())

    assert gt.height == 314
    assert all(
        label[f] == "walking___hopping" for f in [*range(158, 191), *range(219, 238)]
    )
    assert all(label[f] == NO_BIRD for f in [*range(0, 45), *range(191, 219)])
    assert label[157] is None and label[45] == "resting"
    assert label[47] is None and result.conflict_frames == 1  # resting + preening
    # crop frames -1, 0 and 242-245 fall outside the 241 boxed frames
    assert result.clipped_frames == 2 + 4
    assert result.misaligned_detect_tasks == 1
    empty = result.table.filter(pl.col("cage_id") == "1")
    assert empty.height == 314 and (empty["label"] == NO_BIRD).all()
    assert gt.filter(pl.col("frame") == 45).row(0, named=True)["bbox_left"] == 0.1


def _gt_for(cameras: dict[str, list[str]]) -> pl.DataFrame:
    return pl.concat(
        gt_rows(c, "v", labels, "x").drop("split") for c, labels in cameras.items()
    )


def test_split_is_frozen_and_append_only(tmp_path: Path) -> None:
    cameras = {f"c{i}": [b for b in BEHAVIORS if b != "flying"] for i in range(4)}
    cameras |= {f"r{i}": ["resting"] for i in range(3)}

    def run() -> int:
        ratio = {"train": 0.7, "val": 0.15, "test": 0.15}
        return update_split(_gt_for(cameras), tmp_path, ratio, seed=0, n_folds=3)

    assert run() == 1
    v1 = pl.read_parquet(split_path(tmp_path, 1))
    # test: cheapest covering pair (c0, r0); val: one camera covering all
    assert dict(v1.group_by("split").agg(pl.col("camera_id").sort()).iter_rows()) == {
        "test": ["c0", "r0"],
        "val": ["c1"],
        "train": ["c2", "c3", "r1", "r2"],
    }
    assert (tmp_path / "split_v1" / "manifest.json").exists()
    assert run() == 1
    with pytest.raises(ValueError, match="pair"):  # flying only on one camera
        choose_initial_split({"a": {"flying": 1}, "b": {}, "c": {}})

    cameras |= {"new0": ["flying"], "new1": ["resting"]}
    assert run() == 2
    v2 = pl.read_parquet(split_path(tmp_path, 2))
    assert v2.head(v1.height).equals(v1)
    assert set(v2.filter(pl.col("assigned_in_version") == 2)["camera_id"]) == {
        "new0",
        "new1",
    }
    assert v2.filter(camera_id="new0")["split"].item() == "train"  # nobody had it
    gaps = coverage_gaps(v2, camera_behaviors(_gt_for(cameras)))
    assert gaps == {"train": set(), "test": {"flying"}}
    folds1 = pl.read_parquet(cv_folds_path(tmp_path, 1))
    assert (
        pl.read_parquet(cv_folds_path(tmp_path, 2)).head(folds1.height).equals(folds1)
    )


def test_sampling_caps() -> None:
    labeled = (
        ["head_move"] * 10
        + ["preening"] * 4
        + ["resting"] * 30
        + [NO_BIRD] * 20
        + [None] * 3
    )
    frames = pl.concat(
        [
            gt_rows("c", "labeled", labeled, "train"),
            gt_rows("c", "unlabeled", [NO_BIRD] * 7 + [None] * 13, "train"),
            gt_rows("c", "resting_only", ["resting"] * 30, "train"),
            gt_rows("t", "test", labeled, "test"),
        ]
    )
    yolo = select_yolo_frames(frames, stride=5, seed=0)
    counts = dict(
        yolo.filter(pl.col("video_id") == "labeled").group_by("label").len().iter_rows()
    )
    assert counts == {"head_move": 10, "preening": 4, "resting": 10, NO_BIRD: 10}
    assert yolo.filter(pl.col("video_id") == "unlabeled")["frame"].to_list() == [
        0,
        5,
        10,
        15,
    ]
    assert yolo.filter(pl.col("video_id") == "resting_only").height == 6  # stride
    assert yolo.filter(pl.col("split") == "test").height == len(labeled)

    # head_move and preening 1 clip each, so train's 6 resting windows (two
    # cages) are capped to 1 across the whole split
    clips = select_videomae_clips(frames, num_frames=8, stride=8, seed=0)
    train = clips.filter(pl.col("split") == "train")
    assert dict(train.group_by("label").len().iter_rows()) == {
        "head_move": 1,
        "preening": 1,
        "resting": 1,
    }
    assert (
        clips.filter(pl.col("split") == "test", pl.col("label") == "resting").height
        == 3
    )


def test_hash_ignores_ids_and_timestamps() -> None:
    detect = [detect_task("c", "v", "0", [(1, 5)], n=10)]
    classify = [classify_task("c", "v", "0", [("resting", 1, 5)])]
    renamed = copy.deepcopy(detect)
    renamed[0]["annotations"][0]["result"][0]["id"] = "other"
    renamed[0]["updated_at"] = renamed[0]["annotations"][0]["updated_at"] = "2027-01-01"
    assert annotation_hash(detect, classify) == annotation_hash(renamed, classify)
    relabeled = classify_task("c", "v", "0", [("preening", 1, 5)])
    assert annotation_hash(detect, classify) != annotation_hash(detect, [relabeled])

    two = [
        detect_task("c", "v", "0", [(1, 5), (8, 9)], n=10),
        detect_task("c", "v", "1", [(2, 3)], n=10),
    ]
    reordered = copy.deepcopy(two[::-1])
    reordered[1]["annotations"][0]["result"].reverse()
    assert annotation_hash(two, classify) == annotation_hash(reordered, classify)


def test_rejects_non_interpolated_detect_export() -> None:
    task = detect_task("c", "v", "0", [(1, 1)], n=10)
    task["annotations"][0]["result"][0]["value"]["sequence"].append(
        {"frame": 5, "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    )
    with pytest.raises(ValueError, match="interpolate"):
        parse_detect_task(task)


def test_prepare_skips_and_recomputes_on_stride_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "test")
    detect, classify = tmp_path / "detect.json", tmp_path / "classify.json"
    # four unlabeled cameras: test a+b, val c, train d
    tasks = [detect_task(c, "v", "0", [(1, 20)], n=30) for c in "abcd"]
    detect.write_text(json.dumps(tasks))
    classify.write_text("[]")

    def prepare(extra: str = "") -> Path:
        config = tmp_path / "config.toml"
        lines = [
            f'path_to_base_output = "{tmp_path / "out"}"'
            if line.startswith("path_to_base_output")
            else line
            for line in CONFIG.read_text().splitlines()
        ]
        config.write_text("\n".join(lines) + "\n" + extra)
        return run_prepare(
            load_config(config), detect_export=detect, classify_export=classify
        )

    first = prepare()
    assert prepare() == first
    third = prepare("[prepare]\nyolo_uniform_stride = 3\n")
    assert third != first
    yolo = pl.read_parquet(third / "yolo_frames.parquet").filter(split="train")
    frames = yolo["frame"].to_list()
    assert frames == list(range(0, 30, 3))


def test_summary_counts() -> None:
    frames = pl.concat(
        [
            # stretches: resting(0-2), no_bird(3), resting(5-6); frame 4 is a hole
            gt_rows(
                "a", "v1", ["resting"] * 3 + [NO_BIRD, None] + ["resting"] * 2, "train"
            ),
            gt_rows("b", "v2", ["preening", "preening", NO_BIRD], "train"),
            gt_rows("c", "v3", [NO_BIRD, None, None], "train"),  # no Classify task
        ]
    )
    rows = {r["label"]: r for r in summarize(frames).iter_rows(named=True)}
    assert {
        k: (v["frames"], v["stretches"], v["cameras"]) for k, v in rows.items()
    } == {
        "preening": (2, 1, 1),
        "resting": (5, 2, 1),
        NO_BIRD: (3, 3, 3),
        "total": (10, 6, 3),
    }
    assert rows["total"]["labeling_hole_frames"] == 1
    assert rows["total"]["unlabeled_video_frames"] == 2
    assert rows["resting"]["frac_of_split"] == 5 / 10
