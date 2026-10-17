"""Balanced frame selection for YOLO and clip selection for VideoMAE.

Resting is 87% of labeled frames, so train and val keep every active
(non-resting) frame and cap resting and ``no_bird`` at the cage video's most
common active behavior count. Test keeps everything.
"""

import polars as pl

from bird_cv.core.labels import ACTIVE_BEHAVIORS, BEHAVIORS, NO_BIRD, RESTING

CAGE = ["camera_id", "video_id", "cage_id"]

YOLO_FRAMES_SCHEMA = {
    "camera_id": pl.String,
    "video_id": pl.String,
    "cage_id": pl.String,
    "frame": pl.Int64,
    # Ground truth class, or null for a labeling hole (unlabeled video).
    "label": pl.String,
    "split": pl.String,
    # Ground truth box, as in GROUND_TRUTH_SCHEMA; null on no_bird frames.
    "bbox_left": pl.Float64,
    "bbox_top": pl.Float64,
    "bbox_width": pl.Float64,
    "bbox_height": pl.Float64,
}
"""Canonical cage-video frames selected for YOLO, one row per frame."""

VIDEOMAE_CLIPS_SCHEMA = {
    "camera_id": pl.String,
    "video_id": pl.String,
    "cage_id": pl.String,
    "label": pl.String,
    "split": pl.String,
    # Inclusive canonical frames, inside one labeled stretch. Shorter than
    # num_frames only when the stretch is; the consumer pads those.
    "frame_begin": pl.Int64,
    "frame_end": pl.Int64,
}
"""VideoMAE training clips, one row per clip."""


def _active_cap(counts: pl.Expr) -> pl.Expr:
    """Per-cage max of ``counts`` over active-behavior rows (0 if there are none)."""
    return (
        counts.filter(pl.col("label").is_in(ACTIVE_BEHAVIORS))
        .max()
        .over(CAGE)
        .fill_null(0)
    )


def select_yolo_frames(frames: pl.DataFrame, stride: int, seed: int) -> pl.DataFrame:
    """Select YOLO frames from ground truth with a ``split`` column.

    Train and val, per cage video: every active-behavior frame; resting and
    ``no_bird`` each randomly capped at the video's most common active
    behavior count; labeling holes dropped. A cage video with no behavior
    labels at all gets every ``stride``-th canonical frame instead. Test
    keeps every frame.

    Args:
        frames: Ground truth rows with a ``split`` column.
        stride: Frame stride for cage videos with no behavior labels.
        seed: Random seed for the capped samples.

    Returns:
        Selected frames in :data:`YOLO_FRAMES_SCHEMA`.
    """
    frames = frames.sort([*CAGE, "frame"])
    test = frames.filter(pl.col("split") == "test")
    fit = frames.filter(pl.col("split") != "test").with_columns(
        labeled=pl.col("label").is_in(BEHAVIORS).any().over(CAGE),
        count=pl.len().over([*CAGE, "label"]),
        rank=pl.int_range(pl.len()).shuffle(seed=seed).over([*CAGE, "label"]),
    )
    fit = fit.with_columns(cap=_active_cap(pl.col("count")))
    keep = pl.when(~pl.col("labeled")).then(pl.col("frame") % stride == 0)
    keep = keep.when(pl.col("label").is_in(ACTIVE_BEHAVIORS)).then(True)
    keep = keep.when(pl.col("label").is_in([RESTING, NO_BIRD])).then(
        pl.col("rank") < pl.col("cap")
    )
    selected = fit.filter(keep.otherwise(False))
    return (
        pl.concat(
            [
                selected.select(list(YOLO_FRAMES_SCHEMA)),
                test.select(list(YOLO_FRAMES_SCHEMA)),
            ]
        )
        .sort([*CAGE, "frame"])
        .cast(YOLO_FRAMES_SCHEMA)
    )


def labeled_stretches(frames: pl.DataFrame) -> pl.DataFrame:
    """Contiguous same-behavior runs of canonical frames on each cage.

    Args:
        frames: Ground truth rows with a ``split`` column.

    Returns:
        One row per stretch of a behavior (``no_bird`` and holes excluded),
        with ``frame_begin`` and ``frame_end`` inclusive.
    """
    runs = frames.sort([*CAGE, "frame"]).with_columns(
        run=(
            (pl.col("frame").diff() != 1)
            | pl.col("label").ne_missing(pl.col("label").shift())
        )
        .fill_null(True)
        .cum_sum()
        .over(CAGE)
    )
    return (
        runs.filter(pl.col("label").is_in(BEHAVIORS))
        .group_by([*CAGE, "run"])
        .agg(
            pl.col("label").first(),
            pl.col("split").first(),
            frame_begin=pl.col("frame").min(),
            frame_end=pl.col("frame").max(),
        )
        .drop("run")
    )


def select_videomae_clips(
    frames: pl.DataFrame, num_frames: int, stride: int, seed: int
) -> pl.DataFrame:
    """Slice labeled stretches into VideoMAE clips, capping resting in train/val.

    Each stretch yields windows of ``num_frames`` starting every ``stride``
    frames (as in :func:`bird_cv.preprocessing.annotations_to_mae.process_item`);
    a stretch shorter than ``num_frames`` yields one clip covering it. In
    train and val, resting clips are randomly capped per cage video at its
    most common active behavior's clip count.

    Args:
        frames: Ground truth rows with a ``split`` column.
        num_frames: Frames per clip.
        stride: Frames between consecutive clip starts.
        seed: Random seed for the resting cap.

    Returns:
        Selected clips in :data:`VIDEOMAE_CLIPS_SCHEMA`.
    """
    clips = (
        labeled_stretches(frames)
        .with_columns(
            frame_begin=pl.int_ranges(
                pl.col("frame_begin"),
                pl.max_horizontal(
                    pl.col("frame_end") - num_frames + 2, pl.col("frame_begin") + 1
                ),
                stride,
            )
        )
        .explode("frame_begin")
        .with_columns(
            frame_end=pl.min_horizontal(
                pl.col("frame_begin") + num_frames - 1, pl.col("frame_end")
            )
        )
        .sort([*CAGE, "frame_begin"])
        .with_columns(
            count=pl.len().over([*CAGE, "label"]),
            rank=pl.int_range(pl.len()).shuffle(seed=seed).over([*CAGE, "label"]),
        )
        .with_columns(cap=_active_cap(pl.col("count")))
    )
    keep = (
        (pl.col("split") == "test")
        | (pl.col("label") != RESTING)
        | (pl.col("rank") < pl.col("cap"))
    )
    return (
        clips.filter(keep)
        .select(list(VIDEOMAE_CLIPS_SCHEMA))
        .cast(VIDEOMAE_CLIPS_SCHEMA)
    )


def clip_frames(clips: pl.DataFrame) -> pl.DataFrame:
    """Unique canonical frames covered by clips, with their label and split."""
    return (
        clips.with_columns(frame=pl.int_ranges("frame_begin", pl.col("frame_end") + 1))
        .explode("frame")
        .select(*CAGE, "frame", "label", "split")
        .unique()
    )
