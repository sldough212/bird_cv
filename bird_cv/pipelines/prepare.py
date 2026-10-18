"""Prepare training data: snapshot → ground truth → split → summary → sampling.

Each stage skips itself when its outputs already exist. Outputs go where
:class:`bird_cv.core.layout.Layout` says (snapshots, splits, prepared ground
truth and sampling); the run finishes by pointing ``prepared/current.json``
at its output, which every later stage reads.

``cfghash`` hashes every config value sampling reads, so changing one
recomputes the summary and selections instead of reusing stale ones.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

import polars as pl

from bird_cv.core.layout import Layout, Prepared, snapshot_files
from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.data.ground_truth import (
    GT_VERSION,
    build_ground_truth,
    standard_frame_counts,
)
from bird_cv.data.sampling import clip_frames, select_videomae_clips, select_yolo_frames
from bird_cv.data.snapshot import create_snapshot, export_annotations
from bird_cv.data.splits import update_split
from bird_cv.data.summary import log_summary, summarize, with_split
from bird_cv.pipelines.config import BirdCVConfig, load_config, time

logger = logging.getLogger(__name__)


def sampling_config(cfg: BirdCVConfig) -> dict[str, int]:
    """Every config value frame and clip selection reads.

    Args:
        cfg: Loaded pipeline config.

    Returns:
        The values, keyed by name.
    """
    return {
        "yolo_uniform_stride": cfg.prepare.yolo_uniform_stride,
        "videomae_clip_stride": cfg.prepare.videomae_clip_stride,
        "num_frames": clip_length(cfg),
        "seed": cfg.prepare.split_seed,
    }


def clip_length(cfg: BirdCVConfig) -> int:
    """Frames per classifier clip: ``[models.classifier] options.num_frames``.

    Raises:
        ValueError: If the classifier table doesn't set it.
    """
    num_frames = cfg.models.classifier.options.get("num_frames")
    if not isinstance(num_frames, int):
        raise ValueError(
            "[models.classifier] options.num_frames must be set: it is the clip "
            "length prepare samples and the classifier trains on"
        )
    return num_frames


def config_hash(values: dict[str, int]) -> str:
    """Short, stable hash of a config dict, for naming output directories."""
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:8]


def run_prepare(
    cfg: BirdCVConfig,
    hostname: str = "localhost",
    detect_export: Path | None = None,
    classify_export: Path | None = None,
) -> Path:
    """Run every prepare stage and return the sampling output directory.

    Args:
        cfg: Loaded pipeline config.
        hostname: Label Studio host, used only when exporting.
        detect_export: Existing Detect export to snapshot instead of
            exporting from Label Studio. Must be given with
            ``classify_export``.
        classify_export: Existing Classify export to snapshot.

    Returns:
        The sampling directory (:meth:`Layout.sampling_dir`), now also the
        one ``prepared/current.json`` points at.
    """
    layout = Layout(cfg.path_to_base_output)
    pcfg = cfg.prepare

    # Snapshot
    if detect_export is None or classify_export is None:
        detect_export, classify_export = export_annotations(
            host=hostname,
            port=8080,
            api_key=cfg.api_key,
            detect_project_name=cfg.detect_project_name,
            classify_project_name=cfg.classify_project_name,
            output_dir=layout.annotation_exports(time),
        )
    snapshot_hash, snapshot_dir = create_snapshot(
        detect_export,
        classify_export,
        layout.snapshots,
        snapshot_dir_for=layout.snapshot,
    )

    # Ground truth (built before the split, which needs each camera's behaviors)
    gt_path = layout.ground_truth(snapshot_hash, GT_VERSION)
    stats_path = layout.ground_truth_stats(snapshot_hash, GT_VERSION)
    if gt_path.exists():
        logger.info("Ground truth exists: %s", gt_path)
        ground_truth = pl.read_parquet(gt_path)
    else:
        detect_path, classify_path = snapshot_files(snapshot_dir)
        result = build_ground_truth(
            json.loads(detect_path.read_text()),
            json.loads(classify_path.read_text()),
            standard_counts=standard_frame_counts(cfg.path_to_raw_videos),
        )
        ground_truth = result.table
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        ground_truth.write_parquet(gt_path)
        stats_path.write_text(json.dumps(result.counts(), indent=2))

    # Split
    version = update_split(
        ground_truth,
        layout.splits,
        split_ratio=pcfg.split_ratio,
        seed=pcfg.split_seed,
        n_folds=pcfg.cv_folds,
    )
    split = pl.read_parquet(layout.split(version))
    frames = with_split(ground_truth, split)

    # Summary + sampling, keyed by every config value sampling reads
    sampling_cfg = sampling_config(cfg)
    prepared = Prepared(
        snapshot_hash=snapshot_hash,
        gt_version=GT_VERSION,
        split_version=version,
        sampling_hash=config_hash(sampling_cfg),
    )
    out_dir = layout.sampling_dir(prepared)
    summary_path = layout.split_summary(prepared)
    yolo_path = layout.yolo_frames(prepared)
    clips_path = layout.videomae_clips(prepared)
    if all(p.exists() for p in (summary_path, yolo_path, clips_path)):
        logger.info("Sampled data exists: %s", out_dir)
    else:
        yolo = select_yolo_frames(
            frames, stride=pcfg.yolo_uniform_stride, seed=pcfg.split_seed
        )
        clips = select_videomae_clips(
            frames,
            num_frames=clip_length(cfg),
            stride=pcfg.videomae_clip_stride,
            seed=pcfg.split_seed,
        )
        sampled = [
            summarize(
                table.filter(pl.col("split") != "test").with_columns(
                    pl.col("split") + f"_{kind}_sampled"
                )
            )
            for kind, table in (("yolo", yolo), ("videomae", clip_frames(clips)))
        ]
        out_dir.mkdir(parents=True, exist_ok=True)
        yolo.write_parquet(yolo_path)
        clips.write_parquet(clips_path)
        pl.concat([summarize(frames), *sampled]).write_parquet(summary_path)

    counts = json.loads(stats_path.read_text()) if stats_path.exists() else None
    log_summary(pl.read_parquet(summary_path), counts=counts)
    write_manifest(
        new_manifest(
            "prepare",
            snapshot_hash=snapshot_hash,
            split_version=version,
            config={"prepare": asdict(pcfg), "sampling": sampling_cfg},
            inputs={
                "snapshot": str(snapshot_dir),
                "ground_truth": str(gt_path),
                "split": str(layout.split(version)),
                "cv_folds": str(layout.cv_folds(version)),
            },
        ),
        out_dir,
    )
    layout.write_current(prepared)
    logger.info("Prepared data in %s", out_dir)
    # Last stdout line: the id later stages take as --prepared
    print(f"PREPARED={prepared.id}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data.")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file (e.g. configs/config.toml)",
    )
    parser.add_argument(
        "--hostname", default="localhost", help="Label Studio host (node IP)"
    )
    parser.add_argument(
        "--detect-export", type=Path, help="Use this Detect export, skip Label Studio"
    )
    parser.add_argument(
        "--classify-export",
        type=Path,
        help="Use this Classify export, skip Label Studio",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_prepare(
        load_config(args.config_path),
        hostname=args.hostname,
        detect_export=args.detect_export,
        classify_export=args.classify_export,
    )
