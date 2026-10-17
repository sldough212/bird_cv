"""Frozen, append-only camera split and camera cross-validation folds.

See ``docs/adr/0001-split-by-camera.md``. The first version is drawn once
with a fixed seed so that every behavior is in train and test (val best
effort). Later versions only append newly labeled cameras; a camera's split
and fold never change once assigned.
"""

import logging
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from bird_cv.core.labels import BEHAVIORS
from bird_cv.core.schemas import CV_FOLD_SCHEMA, SPLIT_SCHEMA, SPLITS

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 1000
"""Seeded shuffles tried when searching for a split covering every behavior."""

_SPLIT_FILE = re.compile(r"split_v(\d+)\.parquet")


def split_path(splits_dir: Path, version: int) -> Path:
    """Location of split version ``version``."""
    return splits_dir / f"split_v{version}.parquet"


def cv_folds_path(splits_dir: Path, version: int) -> Path:
    """Location of the CV folds belonging to split version ``version``."""
    return splits_dir / f"cv_folds_v{version}.parquet"


def latest_split_version(splits_dir: Path) -> int | None:
    """Highest split version written in ``splits_dir``, or None if there is none."""
    versions = [
        int(m.group(1))
        for p in splits_dir.glob("split_v*.parquet")
        if (m := _SPLIT_FILE.fullmatch(p.name))
    ]
    return max(versions, default=None)


def camera_behaviors(ground_truth: pl.DataFrame) -> dict[str, set[str]]:
    """Behaviors present on each camera in a ground truth table.

    Args:
        ground_truth: Table with ``camera_id`` and ``label`` columns.

    Returns:
        The set of behaviors (not ``no_bird``) on each camera; every camera
        in the table is a key, including cameras with no behavior labels.
    """
    behaviors: dict[str, set[str]] = {
        c: set() for c in ground_truth.get_column("camera_id").unique()
    }
    present = (
        ground_truth.filter(pl.col("label").is_in(BEHAVIORS))
        .select("camera_id", "label")
        .unique()
    )
    for camera_id, label in present.iter_rows():
        behaviors[camera_id].add(label)
    return behaviors


def _covered(cameras: list[str], behaviors: Mapping[str, set[str]]) -> set[str]:
    return set().union(*(behaviors[c] for c in cameras))


def draw_initial_split(
    behaviors: Mapping[str, set[str]], split_ratio: dict[str, float], seed: int
) -> dict[str, list[str]]:
    """Draw the first camera split, requiring every behavior in train and test.

    Shuffles cameras the same way as
    :func:`bird_cv.preprocessing.get_split_guidance.split_camera_data`
    (permute, then cut at ``round(ratio * n)``), trying seeds ``seed``,
    ``seed + 1``, ... up to :data:`MAX_ATTEMPTS`. The first draw with every
    behavior in all three splits wins; otherwise the first draw with every
    behavior in train and test and the most behaviors in val; otherwise (no
    draw covers train and test) the draw covering the most, with a warning.

    Args:
        behaviors: Behaviors present on each camera.
        split_ratio: Fraction of cameras for ``"train"``, ``"val"``, ``"test"``.
        seed: Base random seed.

    Returns:
        Cameras in each split.
    """
    cameras = sorted(behaviors)
    n = len(cameras)
    n_train = round(split_ratio["train"] * n)
    n_val = round(split_ratio["val"] * n)
    target = _covered(cameras, behaviors)

    best: tuple[tuple[int, int], dict[str, list[str]]] | None = None
    for attempt in range(MAX_ATTEMPTS):
        indices = np.random.RandomState(seed + attempt).permutation(n)
        parts = np.split(indices, [n_train, n_train + n_val])
        draw = {
            split: sorted(cameras[i] for i in part)
            for split, part in zip(SPLITS, parts)
        }
        cover = {s: len(_covered(draw[s], behaviors) & target) for s in SPLITS}
        score = (min(cover["train"], cover["test"]), cover["val"])
        if score == (len(target), len(target)):
            return draw
        if best is None or score > best[0]:
            best = (score, draw)

    assert best is not None
    if best[0][0] < len(target):
        logger.warning(
            "No split within %d attempts has every behavior in train and test",
            MAX_ATTEMPTS,
        )
    return best[1]


def _append_cameras(
    split: pl.DataFrame,
    new_behaviors: Mapping[str, set[str]],
    all_behaviors: Mapping[str, set[str]],
    split_ratio: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign new cameras without moving existing ones.

    New cameras are shuffled with ``seed``. Each goes to train if it has a
    behavior both train and test lack, else test if test lacks one of its
    behaviors, else train, else val (same test), else the split furthest
    below its target ratio.
    """
    members = {
        s: split.filter(pl.col("split") == s).get_column("camera_id").to_list()
        for s in SPLITS
    }
    new_cameras = sorted(new_behaviors)
    order = np.random.RandomState(seed).permutation(len(new_cameras))
    assigned = {}
    for i in order:
        camera = new_cameras[i]
        missing = {
            s: new_behaviors[camera] - _covered(members[s], all_behaviors)
            for s in SPLITS
        }
        choice = next(
            (
                s
                for s, lacks in (
                    ("train", missing["train"] & missing["test"]),
                    ("test", missing["test"]),
                    ("train", missing["train"]),
                    ("val", missing["val"]),
                )
                if lacks
            ),
            None,
        )
        if choice is None:
            total = sum(len(m) for m in members.values()) + 1
            choice = max(SPLITS, key=lambda s: split_ratio[s] - len(members[s]) / total)
        members[choice].append(camera)
        assigned[camera] = choice
    return assigned


def coverage_gaps(
    split: pl.DataFrame, behaviors: Mapping[str, set[str]]
) -> dict[str, set[str]]:
    """Behaviors present overall but missing from train or test.

    Args:
        split: Split assignment in :data:`~bird_cv.core.schemas.SPLIT_SCHEMA`.
        behaviors: Behaviors present on each camera.

    Returns:
        For ``"train"`` and ``"test"``, the behaviors they lack (empty sets
        when both cover everything).
    """
    target = _covered(list(behaviors), behaviors)
    return {
        s: target
        - _covered(
            [
                c
                for c in split.filter(pl.col("split") == s)["camera_id"]
                if c in behaviors
            ],
            behaviors,
        )
        for s in ("train", "test")
    }


def assign_folds(
    behaviors: Mapping[str, set[str]], n_folds: int, seed: int
) -> dict[str, int]:
    """Assign cameras to camera-CV folds (each camera is tested in one fold).

    Cameras are dealt round-robin after a seeded shuffle; of
    :data:`MAX_ATTEMPTS` shuffles, the one whose test folds together cover
    the most (fold, behavior) pairs wins.

    Args:
        behaviors: Behaviors present on each camera.
        n_folds: Number of folds.
        seed: Base random seed.

    Returns:
        Fold of each camera.
    """
    cameras = sorted(behaviors)
    best: tuple[int, dict[str, int]] | None = None
    for attempt in range(MAX_ATTEMPTS):
        indices = np.random.RandomState(seed + attempt).permutation(len(cameras))
        folds = {cameras[i]: pos % n_folds for pos, i in enumerate(indices)}
        score = sum(
            len(_covered([c for c, f in folds.items() if f == k], behaviors))
            for k in range(n_folds)
        )
        if best is None or score > best[0]:
            best = (score, folds)
    assert best is not None
    return best[1]


def update_split(
    ground_truth: pl.DataFrame,
    splits_dir: Path,
    split_ratio: dict[str, float],
    seed: int,
    n_folds: int,
) -> int:
    """Return the split version covering every camera, writing a new one if needed.

    With no split yet, draws version 1. If the ground truth has cameras the
    latest version lacks, writes version ``n + 1``: the old rows unchanged
    plus the new cameras (split and CV fold). Otherwise writes nothing.

    Args:
        ground_truth: Table with ``camera_id`` and ``label`` columns.
        splits_dir: Directory holding ``split_v{n}.parquet`` and
            ``cv_folds_v{n}.parquet``.
        split_ratio: Fraction of cameras for ``"train"``, ``"val"``, ``"test"``.
        seed: Base random seed.
        n_folds: Number of camera-CV folds.

    Returns:
        The split version to use.
    """
    behaviors = camera_behaviors(ground_truth)
    latest = latest_split_version(splits_dir)

    if latest is None:
        version = 1
        draw = draw_initial_split(behaviors, split_ratio, seed)
        split = pl.DataFrame(
            [
                {"camera_id": c, "split": s, "assigned_in_version": version}
                for s in SPLITS
                for c in draw[s]
            ],
            schema=SPLIT_SCHEMA,
        )
        folds = pl.DataFrame(
            list(assign_folds(behaviors, n_folds, seed).items()),
            schema=CV_FOLD_SCHEMA,
            orient="row",
        )
    else:
        old_split = pl.read_parquet(split_path(splits_dir, latest))
        old_folds = pl.read_parquet(cv_folds_path(splits_dir, latest))
        known = set(old_split.get_column("camera_id"))
        new = {c: b for c, b in behaviors.items() if c not in known}
        if not new:
            logger.info("Split v%d already covers every camera", latest)
            return latest
        version = latest + 1
        all_behaviors = {c: behaviors.get(c, set()) for c in known} | new
        assigned = _append_cameras(
            old_split, new, all_behaviors, split_ratio, seed + version
        )
        split = pl.concat(
            [
                old_split,
                pl.DataFrame(
                    [
                        {"camera_id": c, "split": s, "assigned_in_version": version}
                        for c, s in sorted(assigned.items())
                    ],
                    schema=SPLIT_SCHEMA,
                ),
            ]
        )
        fold_sizes = {k: 0 for k in range(n_folds)}
        fold_sizes.update(dict(old_folds.group_by("fold").len().iter_rows()))
        new_folds = []
        for camera in sorted(new):
            fold = min(fold_sizes, key=lambda k: (fold_sizes[k], k))
            fold_sizes[fold] += 1
            new_folds.append((camera, fold))
        folds = pl.concat(
            [old_folds, pl.DataFrame(new_folds, schema=CV_FOLD_SCHEMA, orient="row")]
        )

    for name, missing in coverage_gaps(split, behaviors).items():
        if missing:
            logger.warning("Split v%d: %s lacks %s", version, name, sorted(missing))

    splits_dir.mkdir(parents=True, exist_ok=True)
    split.write_parquet(split_path(splits_dir, version))
    folds.write_parquet(cv_folds_path(splits_dir, version))
    logger.info("Wrote split v%d (%d cameras)", version, split.height)
    return version
