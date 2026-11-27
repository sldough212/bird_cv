"""Camera splits: the frozen, append-only split of the new pipelines, and
the split guidance of the Label Studio pipelines.

See ``docs/adr/0001-split-by-camera.md``. The first frozen version is chosen
once by a deterministic rule (:data:`SPLIT_RULE`) so that every behavior is in
train and test (val best effort). Later versions only append newly labeled
cameras; a camera's split never changes once assigned. Each version writes
``split_v{n}/manifest.json`` with its rule and per-camera behavior frame
counts. :func:`simulate_split_guidance` and :func:`split_guidance_within_camera`
serve the Label Studio upload pipelines.
"""

import logging
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from bird_cv.preprocessing.standardize import video_header
from bird_cv.core.ids import with_raw_video_ids
from bird_cv.core.labels import Behaviors
from bird_cv.core.layout import (
    split_file,
    split_manifest_dir,
    split_version_of,
)
from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.core.schemas import SPLIT_SCHEMA, SPLITS

logger = logging.getLogger(__name__)


def latest_split_version(splits_dir: Path) -> int | None:
    """Highest split version written in ``splits_dir``, or None if there is none."""
    versions = [
        v
        for p in splits_dir.glob("split_v*.parquet")
        if (v := split_version_of(p)) is not None
    ]
    return max(versions, default=None)


def camera_behaviors(
    ground_truth: pl.DataFrame, vocab: Behaviors
) -> dict[str, set[str]]:
    """Behaviors present on each camera in a ground truth table.

    Args:
        ground_truth: Table with ``camera_id`` and ``label`` columns.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        The set of behaviors (not ``no_bird``) on each camera; every camera
        in the table is a key, including cameras with no behavior labels.
    """
    behaviors: dict[str, set[str]] = {
        c: set() for c in ground_truth.get_column("camera_id").unique()
    }
    present = (
        ground_truth.filter(pl.col("label").is_in(vocab.names))
        .select("camera_id", "label")
        .unique()
    )
    for camera_id, label in present.iter_rows():
        behaviors[camera_id].add(label)
    return behaviors


def _covered(cameras: list[str], behaviors: Mapping[str, set[str]]) -> set[str]:
    return set().union(*(behaviors[c] for c in cameras))


def camera_behavior_frames(
    ground_truth: pl.DataFrame, vocab: Behaviors
) -> dict[str, dict[str, int]]:
    """Frames of each behavior on each camera.

    Args:
        ground_truth: Table with ``camera_id`` and ``label`` columns.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        ``{camera_id: {behavior: frames}}``; every camera in the table is a
        key, with only the behaviors it has.
    """
    frames: dict[str, dict[str, int]] = {
        c: {} for c in ground_truth.get_column("camera_id").unique().sort()
    }
    counts = (
        ground_truth.filter(pl.col("label").is_in(vocab.names))
        .group_by("camera_id", "label")
        .len()
        .sort("camera_id", "label")
    )
    for camera_id, label, n in counts.iter_rows():
        frames[camera_id][label] = n
    return frames


def _rank(
    cameras: tuple[str, ...],
    frames: Mapping[str, Mapping[str, int]],
    totals: Mapping[str, int],
) -> tuple[float, int, int, tuple[str, ...]]:
    """Rare-frame cost of holding ``cameras`` out of train (lower is better).

    The cost is the sum over behaviors of the cameras' share of that
    behavior's frames, so taking a rare behavior's only stretches costs more
    than taking a lot of resting. Ties go to fewer frames, then fewer
    cameras, then names.
    """
    held = {b: sum(frames[c].get(b, 0) for c in cameras) for b in totals}
    return (
        sum(held[b] / totals[b] for b in totals),
        sum(held.values()),
        len(cameras),
        cameras,
    )


SPLIT_RULE = (
    "test: the camera pair covering every behavior with the lowest "
    "sum over behaviors of (pair frames / total frames), ties to fewer frames; "
    "val: 1-2 of the remaining cameras covering the most behaviors, same "
    "score; train: the rest, which must keep every behavior"
)
"""How :func:`choose_initial_split` picks cameras, recorded in the manifest."""


def choose_initial_split(
    frames: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Choose the first camera split by rule (see :data:`SPLIT_RULE`).

    Test is the pair of cameras covering every behavior that removes the
    fewest rare-behavior frames from training; val is the 1-2 remaining
    cameras covering as many behaviors as possible, ranked the same way;
    train is everything else. A candidate is only allowed if train keeps
    every behavior. Deterministic: ties go to fewer frames, then names.

    Args:
        frames: Frames of each behavior on each camera.

    Returns:
        Cameras in each split.

    Raises:
        ValueError: If no camera pair covers every behavior while leaving
            every behavior in train.
    """
    cameras = sorted(frames)
    totals: dict[str, int] = {}
    for counts in frames.values():
        for b, n in counts.items():
            totals[b] = totals.get(b, 0) + n
    target = set(totals)

    def covers(group: Iterable[str]) -> set[str]:
        return {b for c in group for b in frames[c]}

    tests = sorted(
        (
            _rank(pair, frames, totals)
            for pair in combinations(cameras, 2)
            if covers(pair) == target
            and covers(c for c in cameras if c not in pair) == target
        ),
    )
    if not tests:
        raise ValueError(
            "No camera pair covers every behavior while leaving every behavior "
            f"in train; behaviors per camera: {dict(frames)}"
        )
    test = tests[0][3]

    rest = [c for c in cameras if c not in test]
    vals = sorted(
        (-len(covers(group)), *_rank(group, frames, totals))
        for size in (1, 2)
        for group in combinations(rest, size)
        if covers(c for c in rest if c not in group) == target
    )
    val = vals[0][4] if vals else ()
    if not vals:
        logger.warning("No val camera leaves every behavior in train; val is empty")
    elif -vals[0][0] < len(target):
        logger.warning("Val lacks %s", sorted(target - covers(val)))
    return {
        "train": [c for c in rest if c not in val],
        "val": list(val),
        "test": list(test),
    }


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


def split_behavior_frames(
    split: pl.DataFrame, frames: Mapping[str, Mapping[str, int]], vocab: Behaviors
) -> dict[str, dict[str, int]]:
    """Frames of each behavior in each split.

    Args:
        split: Split assignment in :data:`~bird_cv.core.schemas.SPLIT_SCHEMA`.
        frames: Frames of each behavior on each camera.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        ``{split: {behavior: frames}}`` for every split and every behavior.
    """
    totals = {s: {b: 0 for b in vocab.names} for s in SPLITS}
    for camera_id, name in split.select("camera_id", "split").iter_rows():
        for b, n in frames.get(camera_id, {}).items():
            totals[name][b] += n
    return totals


def update_split(
    ground_truth: pl.DataFrame,
    splits_dir: Path,
    split_ratio: dict[str, float],
    seed: int,
    vocab: Behaviors,
) -> int:
    """Return the split version covering every camera, writing a new one if needed.

    With no split yet, chooses version 1 with :func:`choose_initial_split`. If the ground truth has cameras the
    latest version lacks, writes version ``n + 1``: the old rows unchanged
    plus the new cameras. Otherwise writes nothing.

    Args:
        ground_truth: Table with ``camera_id`` and ``label`` columns.
        splits_dir: Directory holding ``split_v{n}.parquet``.
        split_ratio: Target fraction of cameras per split, used only when
            appending cameras.
        seed: Base random seed for appending cameras.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        The split version to use.
    """
    behaviors = camera_behaviors(ground_truth, vocab)
    behavior_frames = camera_behavior_frames(ground_truth, vocab)
    latest = latest_split_version(splits_dir)

    if latest is None:
        version = 1
        draw = choose_initial_split(behavior_frames)
        rule = SPLIT_RULE
        split = pl.DataFrame(
            [
                {"camera_id": c, "split": s, "assigned_in_version": version}
                for s in SPLITS
                for c in draw[s]
            ],
            schema=SPLIT_SCHEMA,
        )
    else:
        old_split = pl.read_parquet(split_file(splits_dir, latest))
        known = set(old_split.get_column("camera_id"))
        new = {c: b for c, b in behaviors.items() if c not in known}
        if not new:
            logger.info("Split v%d already covers every camera", latest)
            return latest
        version = latest + 1
        rule = (
            f"v{latest} unchanged; new cameras to train if both train and test "
            "lack one of their behaviors, else to test/train/val if it lacks one, "
            "else to the split furthest below split_ratio"
        )
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

    for name, missing in coverage_gaps(split, behaviors).items():
        if missing:
            logger.warning("Split v%d: %s lacks %s", version, name, sorted(missing))

    splits_dir.mkdir(parents=True, exist_ok=True)
    split.write_parquet(split_file(splits_dir, version))
    write_manifest(
        new_manifest(
            "split",
            split_version=version,
            config={
                "rule": rule,
                "split_ratio": split_ratio,
                "seed": seed,
                "behavior_frames": behavior_frames,
                "split_behavior_frames": split_behavior_frames(
                    split, behavior_frames, vocab
                ),
            },
        ),
        split_manifest_dir(splits_dir, version),
    )
    logger.info("Wrote split v%d (%d cameras)", version, split.height)
    return version


def with_split(ground_truth: pl.DataFrame, split: pl.DataFrame) -> pl.DataFrame:
    """Attach each camera's split to ground truth.

    Args:
        ground_truth: Table with a ``camera_id`` column.
        split: Split assignment in :data:`~bird_cv.core.schemas.SPLIT_SCHEMA`.

    Returns:
        ``ground_truth`` with a ``split`` column; cameras missing from
        ``split`` are dropped.
    """
    return ground_truth.join(
        split.select("camera_id", "split"), on="camera_id", how="inner"
    )


def simulate_split_guidance(videos_path: Path, output_path: Path) -> None:
    """Build a split guidance parquet covering every frame of every video found under videos_path.

    Recursively scans for video files, reads their FPS and frame count via OpenCV,
    and writes a parquet with columns ``video_path``, ``fps``, and ``target_frames``
    (a list of all frame indices for each video). Intended as a stand-in for the
    real split guidance produced by the preprocessing pipeline when all frames
    should be processed.

    Args:
        videos_path: Root directory to search recursively for video files.
        output_path: Destination path for the output parquet file.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    video_path_store, fps_store, target_frames_store = [], [], []
    for p in videos_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_extensions:
            # Determine fps of the video
            fps, frame_count = video_header(p)

            video_path_store.append(str(p))
            fps_store.append(fps)
            target_frames_store.append(list(range(1, frame_count + 1)))

    split_guidance = pl.DataFrame(
        {
            "video_path": video_path_store,
            "fps": fps_store,
            "target_frames": target_frames_store,
        }
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)
    split_guidance.write_parquet(output_path)


def split_guidance_within_camera(
    path_to_guidance: Path,
    split_ratio: dict[str, float],
    output_path: Path,
) -> None:
    """Randomly assign each video in a split guidance table to train/val/test.

    Unlike :func:`split_camera_data`, assignment is drawn independently per
    video rather than grouped by camera, so a single camera's videos can end
    up split across train, val, and test.

    Args:
        path_to_guidance: Path to a split guidance parquet (e.g. produced by
            :func:`simulate_split_guidance`) containing at least a
            ``video_path`` column.
        split_ratio: Mapping of split name (``"train"``, ``"val"``,
            ``"test"``) to the probability a given video is assigned to that
            split. Values should sum to 1.
        output_path: Destination path for the output parquet, with a
            ``split`` column added.

    Returns:
        None. Writes the updated split guidance parquet to ``output_path``.
    """
    split_guidance = pl.read_parquet(path_to_guidance)

    # Extract a camera_id
    split_guidance = split_guidance.with_columns(
        camera_id=with_raw_video_ids(split_guidance)["camera_id"]
    )

    splits = np.random.choice(
        list(split_ratio.keys()),
        p=list(split_ratio.values()),
        replace=True,
        size=split_guidance.height,
    )
    split_guidance = split_guidance.with_columns(split=splits)

    split_guidance = split_guidance.drop("camera_id")
    split_guidance.write_parquet(output_path)
