"""Prepare a dataset: snapshot → relabel → ground truth → split → summary → sampling.

Writes ``datasets/{dataset_id}/`` (see :class:`bird_cv.core.layout.Layout`);
the dataset id is ``--run-id`` (a timestamp if not given). Its manifest's
``key`` records the snapshot hash, ground-truth version and inputs, split
version and sampling hash. If an existing dataset has the same key, the new
id becomes a symlink to it instead of a copy, unless ``--force``.

``sampling_hash`` hashes every config value sampling reads, so changing one
gives a new key instead of reusing stale selections.
"""

import argparse
import hashlib
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.layout import Layout, default_run_id, snapshot_files
from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.preprocessing.check_fps import frame_rate_table, standard_counts
from bird_cv.preprocessing.ground_truth import GT_VERSION, build_ground_truth
from bird_cv.preprocessing.sampling import (
    clip_frames,
    log_summary,
    select_videomae_clips,
    select_yolo_frames,
    summarize,
)
from bird_cv.preprocessing.snapshot import export_annotations, snapshot_exports
from bird_cv.preprocessing.splits import update_split, with_split
from bird_cv.pipelines.config import BirdCVConfig, load_config

logger = logging.getLogger(__name__)


def sampling_config(cfg: BirdCVConfig) -> dict[str, Any]:
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
        "behaviors": asdict(cfg.behaviors),
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


def config_hash(values: dict[str, Any]) -> str:
    """Short, stable hash of a config dict, for naming output directories."""
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:8]


def dataset_key(layout: Layout, dataset_id: str) -> dict[str, Any]:
    """The key prepare recorded for a dataset (equal keys mean equal data)."""
    return read_manifest(layout.dataset_dir(dataset_id)).config["key"]


def matching_dataset(layout: Layout, key: dict[str, Any]) -> str | None:
    """Id of a prepared dataset (not a symlink) whose key is ``key``, if any."""
    for manifest in sorted(layout.datasets.glob("*/manifest.json")):
        if not manifest.parent.is_symlink() and (
            read_manifest(manifest.parent).config.get("key") == key
        ):
            return manifest.parent.name
    return None


def run_prepare(
    cfg: BirdCVConfig,
    hostname: str = "localhost",
    detect_export: Path | None = None,
    classify_export: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> str:
    """Run every prepare stage and return the dataset id.

    Args:
        cfg: Loaded pipeline config.
        hostname: Label Studio host, used only when exporting.
        detect_export: Existing Detect export to use instead of exporting
            from Label Studio. Must be given with ``classify_export``.
        classify_export: Existing Classify export.
        run_id: The dataset id; a timestamp if None.
        force: Write a new dataset even if an existing one has the same key.

    Returns:
        The dataset id (``run_id``). If that dataset exists already, it is
        returned untouched.
    """
    layout = Layout(cfg.path_to_base_output)
    pcfg = cfg.prepare
    dataset_id = default_run_id(run_id)
    out_dir = layout.dataset_dir(dataset_id)
    if (out_dir / "manifest.json").exists():
        logger.info("Dataset %s exists: %s", dataset_id, out_dir)
        print(f"DATASET={dataset_id}")
        return dataset_id

    # Snapshot
    annotations = layout.annotations(dataset_id)
    if detect_export is None or classify_export is None:
        detect_export, classify_export = export_annotations(
            host=hostname,
            port=8080,
            api_key=cfg.api_key,
            detect_project_name=cfg.detect_project_name,
            classify_project_name=cfg.classify_project_name,
            output_dir=annotations,
        )
    snapshot_hash = snapshot_exports(detect_export, classify_export, annotations)

    # Ground truth (built before the split, which needs each camera's behaviors).
    # It also depends on the raw headers its frame counts come from.
    gt_key = {
        "gt_version": GT_VERSION,
        "path_to_raw_videos": str(cfg.path_to_raw_videos),
        "standardized_max_frames": pcfg.standardized_max_frames,
        "standard_fps": cfg.standard_fps,
        "behaviors": asdict(cfg.behaviors),
        "relabel": pcfg.relabel,
        "relabel_options": pcfg.relabel_options,
    }
    detect_path, classify_path = snapshot_files(annotations)
    detect_tasks = json.loads(detect_path.read_text())
    rates = frame_rate_table(
        cfg.path_to_raw_videos,
        detect_tasks,
        cfg.standard_fps,
        max_frames=pcfg.standardized_max_frames,
    )
    result = build_ground_truth(
        detect_tasks,
        pcfg.relabeler()(json.loads(classify_path.read_text())),
        cfg.standard_fps,
        cfg.behaviors,
        standard_counts=standard_counts(rates),
    )
    ground_truth = result.table

    # Split
    version = update_split(
        ground_truth,
        layout.splits,
        split_ratio=pcfg.split_ratio,
        seed=pcfg.split_seed,
        vocab=cfg.behaviors,
        relabel=pcfg.relabel,
    )
    split = pl.read_parquet(layout.split(version))

    # Reuse an identical dataset
    sampling_cfg = sampling_config(cfg)
    # Through JSON, so it compares equal to the key read back from a manifest
    key = json.loads(
        json.dumps(
            {
                "snapshot_hash": snapshot_hash,
                "ground_truth": gt_key,
                "split_version": version,
                "sampling_hash": config_hash(sampling_cfg),
            }
        )
    )
    existing = None if force else matching_dataset(layout, key)
    if existing is not None:
        shutil.rmtree(out_dir)
        out_dir.symlink_to(existing, target_is_directory=True)
        logger.info("Dataset %s matches %s; linked to it", dataset_id, existing)
        print(f"DATASET={dataset_id}")
        return dataset_id

    # Ground truth, split, summary + sampling
    ground_truth.write_parquet(layout.ground_truth(dataset_id))
    counts = result.counts()
    layout.ground_truth_stats(dataset_id).write_text(json.dumps(counts, indent=2))
    split.write_parquet(layout.dataset_split_file(dataset_id))
    frames = with_split(ground_truth, split)
    yolo = select_yolo_frames(
        frames,
        stride=pcfg.yolo_uniform_stride,
        seed=pcfg.split_seed,
        vocab=cfg.behaviors,
    )
    clips = select_videomae_clips(
        frames,
        num_frames=clip_length(cfg),
        stride=pcfg.videomae_clip_stride,
        seed=pcfg.split_seed,
        vocab=cfg.behaviors,
    )
    sampled = [
        summarize(
            table.filter(pl.col("split") != "test").with_columns(
                pl.col("split") + f"_{kind}_sampled"
            ),
            cfg.behaviors,
        )
        for kind, table in (("yolo", yolo), ("videomae", clip_frames(clips)))
    ]
    yolo.write_parquet(layout.yolo_frames(dataset_id))
    clips.write_parquet(layout.videomae_clips(dataset_id))
    summary = pl.concat([summarize(frames, cfg.behaviors), *sampled])
    summary.write_parquet(layout.split_summary(dataset_id))
    log_summary(summary, counts=counts)

    # Written last, so an interrupted prepare is redone next time
    write_manifest(
        new_manifest(
            "prepare",
            snapshot_hash=snapshot_hash,
            split_version=version,
            config={"key": key, "prepare": asdict(pcfg), "sampling": sampling_cfg},
            inputs={
                "detect_export": str(detect_export),
                "classify_export": str(classify_export),
                "split": str(layout.split(version)),
            },
        ),
        out_dir,
    )
    logger.info("Prepared dataset %s in %s", dataset_id, out_dir)
    print(f"DATASET={dataset_id}")
    return dataset_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a dataset.")
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
    parser.add_argument("--run-id", default=None, help="Dataset (pipeline run) id")
    parser.add_argument(
        "--force", action="store_true", help="Don't reuse a matching dataset"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_prepare(
        load_config(
            args.config_path,
            require_api_key=args.detect_export is None or args.classify_export is None,
        ),
        hostname=args.hostname,
        detect_export=args.detect_export,
        classify_export=args.classify_export,
        run_id=args.run_id,
        force=args.force,
    )
