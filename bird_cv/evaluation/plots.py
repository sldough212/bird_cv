"""Plots of an end-to-end test run: behavior timelines, confusion matrix,
per-class scores and stage timings.

    python -m bird_cv.evaluation.plots CONFIG --run-id RUN_ID [--prepared ID]

Writes PNGs to ``evaluation/test/{run_id}/plots/``.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from bird_cv.core.labels import CLASSES
from bird_cv.core.layout import Layout
from bird_cv.evaluation.e2e import (
    join_predictions,
    load_ground_truth,
    load_predictions,
)
from bird_cv.evaluation.metrics import frame_metrics
from bird_cv.pipelines.config import load_config


def class_colors() -> dict[str | None, str]:
    """A color per class, plus ``None`` (no prediction / labeling hole)."""
    cmap = plt.get_cmap("tab10")
    colors: dict[str | None, str] = {
        c: matplotlib.colors.to_hex(cmap(i)) for i, c in enumerate(CLASSES[:-1])
    }
    return colors | {"no_bird": "#d9d9d9", None: "#ffffff"}


def _runs(
    frames: np.ndarray, labels: list[str | None]
) -> list[tuple[int, int, str | None]]:
    """Contiguous runs of one label as ``(start frame, length, label)``."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if (
            i == len(labels)
            or labels[i] != labels[start]
            or frames[i] != frames[i - 1] + 1
        ):
            runs.append(
                (
                    int(frames[start]),
                    int(frames[i - 1] - frames[start] + 1),
                    labels[start],
                )
            )
            start = i
    return runs


def plot_timeline(joined: pl.DataFrame, path: Path) -> None:
    """One video's prediction (top) and truth (bottom) strip per cage.

    Args:
        joined: :func:`join_predictions` rows of a single video.
        path: PNG to write.
    """
    colors = class_colors()
    cages = sorted(joined["cage_id"].unique().to_list(), key=int)
    fig, ax = plt.subplots(figsize=(14, 0.9 * len(cages) + 1.2))
    for row, cage in enumerate(cages):
        cage_rows = joined.filter(pl.col("cage_id") == cage).sort("frame")
        frames = cage_rows["frame"].to_numpy()
        for offset, col in ((0.42, "pred_label"), (0.0, "label")):
            for start, length, label in _runs(frames, cage_rows[col].to_list()):
                hole = col == "label" and label is None
                ax.broken_barh(
                    [(start, length)],
                    (row + offset, 0.38),
                    facecolors=colors.get(label, "#000000"),
                    edgecolor="#555555" if hole or label is None else "none",
                    hatch="////" if hole else None,
                    linewidth=0.3,
                )
    ax.set_yticks([r + 0.4 for r in range(len(cages))], [f"cage {c}" for c in cages])
    ax.set_xlabel("frame (top: prediction, bottom: truth)")
    camera, video = joined["camera_id"][0], joined["video_id"][0]
    ax.set_title(f"{camera} / {video}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in CLASSES]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="#555555"))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="#555555", hatch="////"))
    ax.legend(
        handles,
        [*CLASSES, "no prediction", "unlabeled (excluded)"],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(joined: pl.DataFrame, path: Path) -> None:
    """Row-normalized confusion matrix over scored frames, with a column
    for frames that got no prediction.

    Args:
        joined: :func:`join_predictions` rows.
        path: PNG to write.
    """
    scored = joined.filter(pl.col("label").is_not_null())
    cols = [*CLASSES, None]
    counts = np.zeros((len(CLASSES), len(cols)), dtype=int)
    for t, p, n in scored.group_by("label", "pred_label").len().rows():
        counts[CLASSES.index(t), cols.index(p if p in CLASSES else None)] += n
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i, j in np.ndindex(counts.shape):
        if counts[i, j]:
            ax.text(
                j,
                i,
                f"{share[i, j]:.2f}\n{counts[i, j]}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if share[i, j] > 0.5 else "black",
            )
    ax.set_xticks(
        range(len(cols)), [*CLASSES, "no prediction"], rotation=45, ha="right"
    )
    ax.set_yticks(
        range(len(CLASSES)), [f"{c} ({t})" for c, t in zip(CLASSES, totals[:, 0])]
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (frames)")
    ax.set_title("Frame confusion, row-normalized")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_per_class(joined: pl.DataFrame, path: Path) -> None:
    """Precision, recall and F1 per class, labeled with support.

    Args:
        joined: :func:`join_predictions` rows.
        path: PNG to write.
    """
    metrics = frame_metrics(joined)
    per_class = metrics.per_class.with_columns(
        pl.col("precision", "recall", "f1").fill_null(0.0)
    )
    x = np.arange(per_class.height)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, name in enumerate(("precision", "recall", "f1")):
        ax.bar(x + (k - 1) * 0.27, per_class[name].to_numpy(), 0.27, label=name)
    ax.set_xticks(
        x,
        [f"{c}\n(n={s})" for c, s in zip(per_class["label"], per_class["support"])],
        fontsize=8,
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class frame scores (macro-F1 {metrics.macro_f1:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_timings(timings: pl.DataFrame, path: Path) -> None:
    """Total seconds and frames per second for each pipeline stage.

    Args:
        timings: ``TIMINGS_SCHEMA`` rows of an inference run.
        path: PNG to write.
    """
    stages = timings.group_by("stage", maintain_order=True).agg(
        pl.col("seconds").sum(), pl.col("frames").sum()
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.bar(stages["stage"], stages["seconds"] / 60)
    left.set_ylabel("minutes, all videos")
    right.bar(stages["stage"], stages["frames"] / stages["seconds"])
    right.set_ylabel("frames per second (log)")
    right.set_yscale("log")
    gpus = ", ".join(timings["gpu"].unique().to_list())
    fig.suptitle(
        f"Inference stage timings ({timings['video_id'].n_unique()} videos, {gpus})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_test_run(
    config_path: Path, run_id: str, prepared_id: str | None = None
) -> Path:
    """Write every plot for a run's test inference.

    Args:
        config_path: Pipeline config.
        run_id: The pipeline run whose test inference to plot.
        prepared_id: Prepared data to compare against; the run's if ``None``.

    Returns:
        The plots directory.
    """
    cfg = load_config(config_path, require_api_key=False)
    layout = Layout(cfg.path_to_base_output)
    prepared = layout.resolve_prepared(prepared_id, run_id)
    test_run = layout.test_infer_run(run_id)
    pred = load_predictions(test_run.predictions_dir)
    truth = load_ground_truth(
        layout.ground_truth(prepared.snapshot_hash, prepared.gt_version)
    ).filter(pl.col("camera_id").is_in(pred["camera_id"].unique().implode()))
    joined = join_predictions(truth, pred)
    out = layout.evaluation_run("test", run_id) / "plots"
    (out / "timelines").mkdir(parents=True, exist_ok=True)

    plot_confusion(joined, out / "confusion.png")
    plot_per_class(joined, out / "per_class.png")
    for camera, video in (
        joined.filter(pl.col("pred_label").is_not_null())
        .select("camera_id", "video_id")
        .unique()
        .rows()
    ):
        plot_timeline(
            joined.filter(
                (pl.col("camera_id") == camera) & (pl.col("video_id") == video)
            ),
            out / "timelines" / f"{camera}__{video}.png",
        )
    timing_files = sorted(test_run.dir.glob("timings*.parquet"))
    if timing_files:
        plot_timings(
            pl.concat(pl.read_parquet(f, glob=False) for f in timing_files),
            out / "timings.png",
        )
    print(f"Plots: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prepared", default=None)
    args = parser.parse_args()
    plot_test_run(args.config_path, args.run_id, args.prepared)


if __name__ == "__main__":
    main()
