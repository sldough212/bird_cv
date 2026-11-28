"""Build each role's training data inside a prepared dataset.

Uses prepare's sampled list on its frozen split (plus ``no_bird`` clips for
a cage-input classifier), and writes
``datasets/{dataset_id}/{role}/{ModelSpec.dataset_name}/``. Training data whose manifest
matches is not rebuilt.

Usage::

    python bird_cv/pipelines/build_datasets.py CONFIG [--dataset ID | --run-id ID] \\
        [--role ROLE ...] [--processes N]
"""

import argparse
import hashlib
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import ijson
import polars as pl

from bird_cv.core.ids import get_local_path, parse_media_path
from bird_cv.core.layout import Layout, snapshot_files
from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.pipelines.config import BirdCVConfig, ModelSpec, load_config
from bird_cv.pipelines.prepare import config_hash, dataset_key, sampling_config
from bird_cv.preprocessing.sampling import select_no_bird_clips
from bird_cv.training.registry import DatasetMedia, get_trainer

logger = logging.getLogger(__name__)


def export_media(export: Path) -> dict[tuple[str, str, str], Path]:
    """Each task's media path in a Label Studio export.

    Args:
        export: Label Studio JSON export.

    Returns:
        ``{(camera_id, video_id, cage_id): path}`` from each ``data.video``.
    """
    with open(export, "rb") as f:
        return {
            parse_media_path(url): get_local_path(url)
            for url in ijson.items(f, "item.data.video")
        }


def dataset_config(
    role: str,
    spec: ModelSpec,
    key: dict[str, Any],
    split: pl.DataFrame,
    selection: pl.DataFrame,
    media: DatasetMedia,
    media_paths: list[str],
    builder_version: int = 2,
) -> dict[str, Any]:
    """Everything a dataset depends on; equal configs mean an equal dataset.

    Bump ``builder_version`` when a kind's ``build_dataset`` output changes.
    ``media_paths`` are the selected cages' media, hashed so moved media
    rebuild.
    """
    return {
        "builder_version": builder_version,
        "role": role,
        "kind": spec.kind,
        "num_frames": spec.options.get("num_frames"),
        "dataset_key": key,
        "split": [list(r) for r in split.sort("camera_id").iter_rows()],
        "selection_sha256": hashlib.sha256(selection.write_csv().encode()).hexdigest(),
        "media": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(media).items()
            if k != "processes"
        },
        "media_sha256": hashlib.sha256("\n".join(media_paths).encode()).hexdigest(),
    }


def build_datasets(
    cfg: BirdCVConfig,
    dataset_id: str | None = None,
    roles: list[str] | None = None,
    processes: int = 1,
    run_id: str | None = None,
) -> list[Path]:
    """Build (or reuse) the datasets of ``roles`` on the frozen split.

    Args:
        cfg: Pipeline config; its sampling values must be the ones the
            dataset was sampled with.
        dataset_id: The prepared dataset; ``run_id`` if None.
        roles: ``[models]`` roles; all if None.
        processes: Worker processes for the converters.
        run_id: Pipeline run id, the dataset id if ``dataset_id`` is None.

    Returns:
        The training data directories.

    Raises:
        ValueError: If the config's sampling values differ from the
            dataset's, or a role is unknown.
        FileNotFoundError: If a selected cage's media (from the export) is
            missing; raised before any dataset is built.
    """
    layout = Layout(cfg.path_to_base_output)
    dataset_id = layout.resolve_dataset(dataset_id, run_id)
    data_key = dataset_key(layout, dataset_id)
    sampling = sampling_config(cfg)
    if config_hash(sampling) != data_key["sampling_hash"]:
        raise ValueError(
            f"The config's sampling values {sampling} hash to "
            f"{config_hash(sampling)}, not dataset {dataset_id}'s "
            f"{data_key['sampling_hash']}; rerun prepare or restore the config"
        )
    roles = roles or list(cfg.models.roles())
    unknown = set(roles) - set(cfg.models.roles())
    if unknown:
        raise ValueError(f"Unknown roles {sorted(unknown)}")

    ground_truth = pl.read_parquet(layout.ground_truth(dataset_id))
    frozen = pl.read_parquet(layout.dataset_split_file(dataset_id))
    detect_export, classify_export = snapshot_files(layout.annotations(dataset_id))
    media = DatasetMedia(
        detect_export=detect_export,
        classify_export=classify_export,
        standard_fps=cfg.standard_fps,
        processes=processes,
    )

    plans = []
    missing: set[str] = set()
    for role in roles:
        spec = cfg.models.roles()[role]
        trainer = get_trainer(spec.kind)
        paths = export_media(trainer.dataset_export(media, spec))
        split = frozen.select("camera_id", "split")
        selection = pl.read_parquet(trainer.prepared_selection(layout, dataset_id))
        if spec.input == "cage":
            no_bird = select_no_bird_clips(
                ground_truth.join(split, on="camera_id"),
                selection,
                num_frames=spec.options["num_frames"],
                seed=cfg.prepare.split_seed,
                vocab=cfg.behaviors,
            )
            selection = pl.concat([selection, no_bird]).sort(
                "camera_id", "video_id", "cage_id", "frame_begin"
            )
        cages = selection.select("camera_id", "video_id", "cage_id").unique()
        media_paths = []
        for key in cages.iter_rows():
            path = paths.get(key)
            media_paths.append(str(path))
            if path is None or not path.exists():
                missing.add(str(path or key))
        config = dataset_config(
            role, spec, data_key, split, selection, media, sorted(media_paths)
        )
        plans.append((role, spec, trainer, split, selection, config))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected media files are missing, e.g. "
            f"{sorted(missing)[:10]}"
        )

    built = []
    for role, spec, trainer, split, selection, config in plans:
        out_dir = layout.dataset(dataset_id, role, spec.dataset_name)
        name = f"{role}/{spec.dataset_name}"
        if (out_dir / "manifest.json").exists() and read_manifest(
            out_dir
        ).config == config:
            logger.info("Dataset %s is up to date: %s", name, out_dir)
            built.append(out_dir)
            continue
        if out_dir.exists():
            logger.info("Rebuilding stale dataset %s", out_dir)
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        split.write_parquet(layout.dataset_split(dataset_id, role, spec.dataset_name))
        selection.write_parquet(
            layout.dataset_selection(dataset_id, role, spec.dataset_name)
        )
        logger.info("Building dataset %s: %d rows", name, selection.height)
        trainer.build_dataset(
            selection,
            ground_truth,
            spec,
            media,
            out_dir,
            layout.dataset_guidance(dataset_id, role, spec.dataset_name),
        )
        # Written last, so an interrupted build is rebuilt next time
        write_manifest(
            new_manifest(
                "dataset",
                snapshot_hash=data_key["snapshot_hash"],
                split_version=data_key["split_version"],
                config=config,
                inputs={
                    "ground_truth": str(layout.ground_truth(dataset_id)),
                    "detect_export": str(detect_export),
                    "classify_export": str(classify_export),
                },
            ),
            out_dir,
        )
        built.append(out_dir)
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument(
        "--dataset", default=None, help="Dataset id (default: --run-id)"
    )
    parser.add_argument("--role", action="append", dest="roles", help="Repeatable")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--run-id", default=None, help="Pipeline run id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for out_dir in build_datasets(
        load_config(args.config_path, require_api_key=False),
        args.dataset,
        args.roles,
        args.processes,
        args.run_id,
    ):
        print(f"Dataset: {out_dir}")


if __name__ == "__main__":
    main()
