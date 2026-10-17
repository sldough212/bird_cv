"""Prepare training data: snapshot → ground truth → split → summary → sampling.

Each stage skips itself when its outputs already exist. Outputs under
``path_to_base_output``::

    snapshots/{hash}/{detect.json, classify.json, manifest.json}
    splits/{split_v{n}.parquet, cv_folds_v{n}.parquet}
    prepared/{hash}/ground_truth.parquet
    prepared/{hash}/split_v{n}/{split_summary, yolo_frames, videomae_clips}.parquet
    prepared/{hash}/split_v{n}/manifest.json
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import polars as pl

from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.data.ground_truth import build_ground_truth
from bird_cv.data.sampling import clip_frames, select_videomae_clips, select_yolo_frames
from bird_cv.data.snapshot import create_snapshot, export_annotations
from bird_cv.data.splits import cv_folds_path, split_path, update_split
from bird_cv.data.summary import log_summary, summarize, with_split
from bird_cv.pipelines.config import BirdCVConfig, load_config, time

logger = logging.getLogger(__name__)


def run_prepare(
    cfg: BirdCVConfig,
    hostname: str = "localhost",
    detect_export: Path | None = None,
    classify_export: Path | None = None,
) -> Path:
    """Run every prepare stage and return the prepared output directory.

    Args:
        cfg: Loaded pipeline config.
        hostname: Label Studio host, used only when exporting.
        detect_export: Existing Detect export to snapshot instead of
            exporting from Label Studio. Must be given with
            ``classify_export``.
        classify_export: Existing Classify export to snapshot.

    Returns:
        ``prepared/{hash}/split_v{n}``.
    """
    base = cfg.path_to_base_output
    pcfg = cfg.prepare

    # Snapshot
    if detect_export is None or classify_export is None:
        detect_export, classify_export = export_annotations(
            host=hostname,
            port=8080,
            api_key=cfg.api_key,
            detect_project_name=cfg.detect_project_name,
            classify_project_name=cfg.classify_project_name,
            output_dir=base / "annotations" / time,
        )
    snapshot_hash, snapshot_dir = create_snapshot(
        detect_export, classify_export, base / "snapshots"
    )

    # Ground truth (built before the split, which needs each camera's behaviors)
    gt_dir = base / "prepared" / snapshot_hash
    gt_path = gt_dir / "ground_truth.parquet"
    if gt_path.exists():
        logger.info("Ground truth exists: %s", gt_path)
        ground_truth = pl.read_parquet(gt_path)
    else:
        result = build_ground_truth(
            json.loads((snapshot_dir / "detect.json").read_text()),
            json.loads((snapshot_dir / "classify.json").read_text()),
        )
        ground_truth = result.table
        gt_dir.mkdir(parents=True, exist_ok=True)
        ground_truth.write_parquet(gt_path)
        stats = {k: v for k, v in vars(result).items() if k != "table"}
        (gt_dir / "ground_truth_stats.json").write_text(json.dumps(stats, indent=2))

    # Split
    splits_dir = base / "splits"
    version = update_split(
        ground_truth,
        splits_dir,
        split_ratio=pcfg.split_ratio,
        seed=pcfg.split_seed,
        n_folds=pcfg.cv_folds,
    )
    split = pl.read_parquet(split_path(splits_dir, version))
    frames = with_split(ground_truth, split)
    out_dir = gt_dir / f"split_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary_path = out_dir / "split_summary.parquet"
    if not summary_path.exists():
        summarize(frames).write_parquet(summary_path)

    # Sampling
    yolo_path = out_dir / "yolo_frames.parquet"
    clips_path = out_dir / "videomae_clips.parquet"
    if not (yolo_path.exists() and clips_path.exists()):
        yolo = select_yolo_frames(
            frames, stride=pcfg.yolo_uniform_stride, seed=pcfg.split_seed
        )
        clips = select_videomae_clips(
            frames,
            num_frames=cfg.classify.num_frames,
            stride=pcfg.videomae_clip_stride,
            seed=pcfg.split_seed,
        )
        sampled = []
        for kind, table in (("yolo", yolo), ("videomae", clip_frames(clips))):
            fit = table.filter(pl.col("split") != "test")
            sampled.append(
                summarize(fit.with_columns(pl.col("split") + f"_{kind}_sampled"))
            )
        base_rows = pl.read_parquet(summary_path).filter(
            ~pl.col("split").str.ends_with("_sampled")
        )
        pl.concat([base_rows, *sampled]).write_parquet(summary_path)
        yolo.write_parquet(yolo_path)
        clips.write_parquet(clips_path)

    log_summary(
        pl.read_parquet(summary_path),
        counts=json.loads((gt_dir / "ground_truth_stats.json").read_text()),
    )
    write_manifest(
        new_manifest(
            "prepare",
            snapshot_hash=snapshot_hash,
            split_version=version,
            config={
                "prepare": asdict(pcfg),
                "num_frames": cfg.classify.num_frames,
            },
            inputs={
                "snapshot": str(snapshot_dir),
                "split": str(split_path(splits_dir, version)),
                "cv_folds": str(cv_folds_path(splits_dir, version)),
            },
        ),
        out_dir,
    )
    logger.info("Prepared data in %s", out_dir)
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
