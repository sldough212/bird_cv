"""Per-split class counts (``split_summary.parquet``) and a readable log table."""

import logging

import polars as pl

from bird_cv.core.labels import BEHAVIORS, CLASSES
from bird_cv.core.schemas import SPLIT_SUMMARY_SCHEMA, SPLITS

logger = logging.getLogger(__name__)

_CAGE = ("camera_id", "video_id", "cage_id")


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


def summarize(frames: pl.DataFrame) -> pl.DataFrame:
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

    Returns:
        One row per (split, class) present plus a ``"total"`` row per split,
        in :data:`~bird_cv.core.schemas.SPLIT_SUMMARY_SCHEMA`.
    """
    runs = frames.sort([*_CAGE, "frame"]).with_columns(
        new_stretch=(
            (pl.col("frame").diff() != 1)
            | (pl.col("label").ne_missing(pl.col("label").shift()))
        )
        .over(["split", *_CAGE])
        .fill_null(True),
        labeled_video=pl.col("label").is_in(BEHAVIORS).any().over(_CAGE),
    )
    group = {
        "frames": pl.len(),
        "stretches": pl.col("new_stretch").sum(),
        "cameras": pl.col("camera_id").n_unique(),
        "videos": pl.struct("camera_id", "video_id").n_unique(),
        "cages": pl.struct(*_CAGE).n_unique(),
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
    order = {c: i for i, c in enumerate((*CLASSES, "total"))}
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
