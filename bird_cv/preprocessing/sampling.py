"""Balanced frame selection for YOLO and clip selection for VideoMAE.

Resting is 87% of labeled frames, so train and val keep every active
(non-resting) frame and cap resting: YOLO per cage video (resting and
``no_bird`` at the video's most common active behavior count), VideoMAE per
split (resting clips at the split's most common active behavior clip count).
Test keeps everything. Also the per-split class counts
(``split_summary.parquet``) of the sampled data.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

from bird_cv.core.ids import parse_raw_video_path
from bird_cv.core.labels import Behaviors
from bird_cv.core.schemas import (
    CAGE_COLUMNS,
    CAGE_FRAMES_SCHEMA,
    SPLIT_SUMMARY_SCHEMA,
    SPLITS,
    VIDEOMAE_CLIPS_SCHEMA,
    YOLO_FRAMES_SCHEMA,
)

logger = logging.getLogger(__name__)


def _active_cap(counts: pl.Expr, vocab: Behaviors) -> pl.Expr:
    """Per-cage max of ``counts`` over active-behavior rows (0 if there are none)."""
    return (
        counts.filter(pl.col("label").is_in(vocab.active))
        .max()
        .over(CAGE_COLUMNS)
        .fill_null(0)
    )


def select_yolo_frames(
    frames: pl.DataFrame, stride: int, seed: int, vocab: Behaviors
) -> pl.DataFrame:
    """Select YOLO frames from ground truth with a ``split`` column.

    Train and val, per cage video: every active-behavior frame; resting and
    ``no_bird`` each randomly capped at the video's most common active
    behavior count; labeling holes dropped. A cage video with no active
    behavior (unlabeled, or resting only) gets every ``stride``-th canonical
    frame instead, whatever its labels. Test keeps every frame.

    Args:
        frames: Ground truth rows with a ``split`` column.
        stride: Frame stride for cage videos with no active behavior.
        seed: Random seed for the capped samples.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        Selected frames in :data:`YOLO_FRAMES_SCHEMA`.
    """
    frames = frames.sort([*CAGE_COLUMNS, "frame"])
    test = frames.filter(pl.col("split") == "test")
    fit = frames.filter(pl.col("split") != "test").with_columns(
        active=pl.col("label").is_in(vocab.active).any().over(CAGE_COLUMNS),
        count=pl.len().over([*CAGE_COLUMNS, "label"]),
        rank=pl.int_range(pl.len()).shuffle(seed=seed).over([*CAGE_COLUMNS, "label"]),
    )
    fit = fit.with_columns(cap=_active_cap(pl.col("count"), vocab))
    keep = pl.when(~pl.col("active")).then(pl.col("frame") % stride == 0)
    keep = keep.when(pl.col("label").is_in(vocab.active)).then(True)
    keep = keep.when(pl.col("label").is_in([vocab.resting, vocab.no_bird])).then(
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
        .sort([*CAGE_COLUMNS, "frame"])
        .cast(YOLO_FRAMES_SCHEMA)
    )


def select_cage_frames(
    videos: list[Path], split: pl.DataFrame, masks_root: Path, per_video: int
) -> pl.DataFrame:
    """Select cage-detector frames: ``per_video`` per raw video of a split camera.

    Frames spread over the video's legacy one-shot segments: one per
    segment (evenly spaced segments if there are more), cycling through
    them if there are fewer, spaced evenly within each segment.

    Args:
        videos: Raw videos, ``{camera}/{video}.{ext}``.
        split: ``camera_id``, ``split``; videos of other cameras are skipped.
        masks_root: ``{camera}/{video}/`` legacy masks root.
        per_video: Frames per video.

    Returns:
        :data:`CAGE_FRAMES_SCHEMA` rows.

    Raises:
        FileNotFoundError: Listing every split video with no legacy masks.
    """
    split_of = dict(split.select("camera_id", "split").iter_rows())
    rows, missing = [], []
    for video in videos:
        camera_id, video_id = parse_raw_video_path(video)
        if camera_id not in split_of:
            continue
        video_dir = masks_root / camera_id / video_id
        if not (video_dir / "segment_index.json").is_file():
            missing.append(f"{camera_id}/{video_id}")
            continue
        index = json.loads((video_dir / "segment_index.json").read_text())
        segs = sorted(index, key=int)
        if len(segs) > per_video:
            segs = [segs[round(i)] for i in np.linspace(0, len(segs) - 1, per_video)]
        uses = Counter(segs[k % len(segs)] for k in range(per_video))
        for seg, n in uses.items():
            start, end = index[seg]["start"], index[seg]["end"]
            for r in range(n):
                rows.append(
                    {
                        "camera_id": camera_id,
                        "video_id": video_id,
                        "split": split_of[camera_id],
                        "video_path": str(video),
                        # Legacy segment frame i is raw frame i + 1
                        "raw_frame": start + (end - start) * (r + 1) // (n + 1) + 1,
                        "mask_path": str(video_dir / f"{seg}_segmentation.json"),
                    }
                )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} videos have no legacy cage masks under {masks_root}, "
            f"e.g. {missing[:10]}"
        )
    return pl.DataFrame(rows, schema=CAGE_FRAMES_SCHEMA).unique(maintain_order=True)


def labeled_stretches(frames: pl.DataFrame, vocab: Behaviors) -> pl.DataFrame:
    """Contiguous same-behavior runs of canonical frames on each cage.

    Args:
        frames: Ground truth rows with a ``split`` column.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        One row per stretch of a behavior (``no_bird`` and holes excluded),
        with ``frame_begin`` and ``frame_end`` inclusive.
    """
    runs = frames.sort([*CAGE_COLUMNS, "frame"]).with_columns(
        run=(
            (pl.col("frame").diff() != 1)
            | pl.col("label").ne_missing(pl.col("label").shift())
        )
        .fill_null(True)
        .cum_sum()
        .over(CAGE_COLUMNS)
    )
    return (
        runs.filter(pl.col("label").is_in(vocab.names))
        .group_by([*CAGE_COLUMNS, "run"])
        .agg(
            pl.col("label").first(),
            pl.col("split").first(),
            frame_begin=pl.col("frame").min(),
            frame_end=pl.col("frame").max(),
        )
        .drop("run")
    )


def select_videomae_clips(
    frames: pl.DataFrame, num_frames: int, stride: int, seed: int, vocab: Behaviors
) -> pl.DataFrame:
    """Slice labeled stretches into VideoMAE clips, capping resting in train/val.

    Each stretch yields windows of ``num_frames`` starting every ``stride``
    frames (as in :func:`bird_cv.preprocessing.annotations_to_mae.process_item`);
    a stretch shorter than ``num_frames`` yields one clip covering it. In
    train and val, resting clips are capped per split at the split's most
    common active behavior's clip count, taken round-robin across cage
    videos (each video's clips in seeded random order) so every video
    contributes evenly.

    Args:
        frames: Ground truth rows with a ``split`` column.
        num_frames: Frames per clip.
        stride: Frames between consecutive clip starts.
        seed: Random seed for the resting cap.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        Selected clips in :data:`VIDEOMAE_CLIPS_SCHEMA`.
    """
    clips = (
        labeled_stretches(frames, vocab)
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
        .sort([*CAGE_COLUMNS, "frame_begin"])
        .with_columns(
            rank=pl.int_range(pl.len())
            .shuffle(seed=seed)
            .over([*CAGE_COLUMNS, "label"])
        )
    )
    capped = (pl.col("label") == vocab.resting) & (pl.col("split") != "test")
    caps = (
        clips.filter(pl.col("label").is_in(vocab.active))
        .group_by("split", "label")
        .len()
        .group_by("split")
        .agg(cap=pl.col("len").max())
    )
    resting = (
        clips.filter(capped)
        .join(caps, on="split", how="left")
        .with_columns(pl.col("cap").fill_null(0))
        .sort(["split", "rank", *CAGE_COLUMNS])
        .filter(pl.int_range(pl.len()).over("split") < pl.col("cap"))
    )
    return (
        pl.concat(
            [
                clips.filter(~capped).select(list(VIDEOMAE_CLIPS_SCHEMA)),
                resting.select(list(VIDEOMAE_CLIPS_SCHEMA)),
            ]
        )
        .sort([*CAGE_COLUMNS, "frame_begin"])
        .cast(VIDEOMAE_CLIPS_SCHEMA)
    )


def clip_frames(clips: pl.DataFrame) -> pl.DataFrame:
    """Unique canonical frames covered by clips, with their label and split.

    Args:
        clips: Clips in :data:`~bird_cv.core.schemas.VIDEOMAE_CLIPS_SCHEMA`.

    Returns:
        One row per covered (cage, frame) with ``label`` and ``split``, for
        summarizing clip selections like frame selections.
    """
    return (
        clips.with_columns(frame=pl.int_ranges("frame_begin", pl.col("frame_end") + 1))
        .explode("frame")
        .select(*CAGE_COLUMNS, "frame", "label", "split")
        .unique()
    )


def summarize(frames: pl.DataFrame, vocab: Behaviors) -> pl.DataFrame:
    """Count frames, stretches, cameras, videos and cages per split and class.

    A stretch is a run of consecutive canonical frames with the same label
    on one cage. Null labels count only towards the total row: as
    ``unlabeled_video_frames`` in cage videos with no behavior label at all,
    else as ``labeling_hole_frames``. ``frac_of_split`` is relative to the
    split's non-null frames. ``stretches`` is null on ``*_sampled`` splits,
    where runs of sampled frames are not labeled stretches.

    Args:
        frames: Table with ``camera_id``, ``video_id``, ``cage_id``, ``frame``,
            ``label`` and ``split`` columns.
        vocab: The class vocabulary (``[behaviors]`` in config).

    Returns:
        One row per (split, class) present plus a ``"total"`` row per split,
        in :data:`~bird_cv.core.schemas.SPLIT_SUMMARY_SCHEMA`.
    """
    runs = frames.sort([*CAGE_COLUMNS, "frame"]).with_columns(
        new_stretch=(
            (pl.col("frame").diff() != 1)
            | (pl.col("label").ne_missing(pl.col("label").shift()))
        )
        .over(["split", *CAGE_COLUMNS])
        .fill_null(True),
        labeled_video=pl.col("label").is_in(vocab.names).any().over(CAGE_COLUMNS),
    )
    group = {
        "frames": pl.len(),
        "stretches": pl.col("new_stretch").sum(),
        "cameras": pl.col("camera_id").n_unique(),
        "videos": pl.struct("camera_id", "video_id").n_unique(),
        "cages": pl.struct(*CAGE_COLUMNS).n_unique(),
    }
    labeled = runs.filter(pl.col("label").is_not_null())
    per_class = labeled.group_by("split", "label").agg(**group)
    totals = (
        runs.group_by("split")
        .agg(
            frames=pl.col("label").is_not_null().sum(),
            stretches=pl.col("new_stretch").filter(pl.col("label").is_not_null()).sum(),
            cameras=group["cameras"],
            videos=group["videos"],
            cages=group["cages"],
            labeling_hole_frames=(
                pl.col("label").is_null() & pl.col("labeled_video")
            ).sum(),
            unlabeled_video_frames=(
                pl.col("label").is_null() & ~pl.col("labeled_video")
            ).sum(),
        )
        .with_columns(label=pl.lit("total"))
    )
    order = {c: i for i, c in enumerate((*vocab.classes, "total"))}
    split_frames = totals.select("split", split_frames="frames")
    summary = (
        pl.concat([per_class, totals], how="diagonal_relaxed")
        .join(split_frames, on="split")
        .with_columns(
            frac_of_split=pl.col("frames") / pl.col("split_frames"),
            stretches=pl.when(~pl.col("split").str.ends_with("_sampled")).then(
                pl.col("stretches")
            ),
            _order=pl.col("label").replace_strict(order, default=len(order)),
            _split_order=pl.col("split").replace_strict(
                {s: i for i, s in enumerate(SPLITS)}, default=len(SPLITS)
            ),
        )
        .sort("_split_order", "split", "_order")
    )
    return summary.select(
        pl.col(name).cast(dtype) for name, dtype in SPLIT_SUMMARY_SCHEMA.items()
    )


def log_summary(summary: pl.DataFrame, counts: dict[str, int] | None = None) -> str:
    """Log a split summary as a readable table and return the text.

    Args:
        summary: Table in :data:`~bird_cv.core.schemas.SPLIT_SUMMARY_SCHEMA`.
        counts: Extra counts to print under the table, e.g. the ground truth
            conversion's excluded (misaligned) Detect tasks.

    Returns:
        The logged text.
    """
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True
    ):
        text = str(summary.with_columns(pl.col("frac_of_split").round(4)))
    for name, value in (counts or {}).items():
        text += f"\n{name}: {value}"
    logger.info("Split summary:\n%s", text)
    return text
