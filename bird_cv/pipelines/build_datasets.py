"""Build each role's training data from prepared data: frozen split and CV folds.

The frozen split uses prepare's sampled list; each CV fold reruns the
sampling on its :func:`~bird_cv.evaluation.cv.fold_split`. A dataset whose
manifest matches is not rebuilt.

Usage::

    python bird_cv/pipelines/build_datasets.py CONFIG --prepared ID \\
        [--role ROLE ...] [--fold K ...] [--processes N]

Without ``--fold``, builds the frozen split and every fold.
"""

import argparse
import hashlib
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ijson
import polars as pl

from bird_cv.client.utils import get_local_path
from bird_cv.core.ids import parse_media_path
from bird_cv.core.layout import Layout, Prepared, snapshot_files
from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.data.summary import with_split
from bird_cv.evaluation.cv import fold_split
from bird_cv.pipelines.config import BirdCVConfig, load_config
from bird_cv.pipelines.prepare import config_hash, sampling_config

if TYPE_CHECKING:
    from bird_cv.pipelines.config import ModelSpec
    from bird_cv.training.registry import DatasetMedia

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
    spec: "ModelSpec",
    fold: int | None,
    prepared: Prepared,
    split: pl.DataFrame,
    selection: pl.DataFrame,
    media: "DatasetMedia",
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
        "fold": fold,
        "prepared": asdict(prepared),
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
    prepared_id: str | None = None,
    roles: list[str] | None = None,
    folds: list[int] | None = None,
    processes: int = 1,
    run_id: str | None = None,
) -> list[Path]:
    """Build (or reuse) the datasets of ``roles`` on the frozen split and ``folds``.

    Args:
        cfg: Pipeline config; its sampling values must be the ones the
            prepared data was sampled with.
        prepared_id: :attr:`Prepared.id`; the last prepare output (logged)
            if None.
        roles: ``[models]`` roles; all if None.
        folds: CV folds to build, without the frozen split; if None, the
            frozen split and every fold.
        processes: Worker processes for the converters.
        run_id: Pipeline run whose prepared id to use if ``prepared_id`` is None.

    Returns:
        The dataset directories.

    Raises:
        ValueError: If the config's sampling values differ from the prepared
            data's, or a role is unknown.
        FileNotFoundError: If a selected cage's media (from the export) is
            missing; raised before any dataset is built.
    """
    from bird_cv.training.registry import DatasetMedia, get_trainer

    layout = Layout(cfg.path_to_base_output)
    prepared = layout.resolve_prepared(prepared_id, run_id)
    sampling = sampling_config(cfg)
    if config_hash(sampling) != prepared.sampling_hash:
        raise ValueError(
            f"The config's sampling values {sampling} hash to "
            f"{config_hash(sampling)}, not the prepared data's "
            f"{prepared.sampling_hash}; rerun prepare or restore the config"
        )
    roles = roles or list(cfg.models.roles())
    unknown = set(roles) - set(cfg.models.roles())
    if unknown:
        raise ValueError(f"Unknown roles {sorted(unknown)}")

    ground_truth = pl.read_parquet(
        layout.ground_truth(prepared.snapshot_hash, prepared.gt_version)
    )
    cv_folds = pl.read_parquet(layout.cv_folds(prepared.split_version))
    frozen = pl.read_parquet(layout.split(prepared.split_version))
    detect_export, classify_export = snapshot_files(
        layout.snapshot(prepared.snapshot_hash)
    )
    media = DatasetMedia(
        detect_export=detect_export,
        classify_export=classify_export,
        processes=processes,
    )
    targets: list[int | None] = (
        [None, *sorted(cv_folds["fold"].unique().to_list())]
        if folds is None
        else list(folds)
    )

    plans = []
    missing: set[str] = set()
    for role in roles:
        spec = cfg.models.roles()[role]
        trainer = get_trainer(spec.kind)
        paths = export_media(trainer.dataset_export(media))
        for fold in targets:
            if fold is None:
                split = frozen.select("camera_id", "split")
                selection = pl.read_parquet(
                    trainer.prepared_selection(layout, prepared)
                )
            else:
                split = fold_split(
                    cv_folds,
                    fold,
                    n_val_cameras=cfg.evaluate.cv_val_cameras,
                    seed=cfg.prepare.split_seed,
                )
                selection = trainer.select(with_split(ground_truth, split), sampling)
            cages = selection.select("camera_id", "video_id", "cage_id").unique()
            media_paths = []
            for key in cages.iter_rows():
                path = paths.get(key)
                media_paths.append(str(path))
                if path is None or not path.exists():
                    missing.add(str(path or key))
            config = dataset_config(
                role, spec, fold, prepared, split, selection, media, sorted(media_paths)
            )
            plans.append((role, spec, trainer, fold, split, selection, config))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected media files are missing, e.g. "
            f"{sorted(missing)[:10]}"
        )

    built = []
    for role, spec, trainer, fold, split, selection, config in plans:
        out_dir = layout.dataset(prepared, role, spec.kind, fold)
        name = f"{role}/{spec.kind} " + ("frozen" if fold is None else f"fold{fold}")
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
        split.write_parquet(layout.dataset_split(prepared, role, spec.kind, fold))
        selection.write_parquet(
            layout.dataset_selection(prepared, role, spec.kind, fold)
        )
        logger.info("Building dataset %s: %d rows", name, selection.height)
        trainer.build_dataset(
            selection,
            ground_truth,
            spec,
            media,
            out_dir,
            layout.dataset_guidance(prepared, role, spec.kind, fold),
        )
        # Written last, so an interrupted build is rebuilt next time
        write_manifest(
            new_manifest(
                "dataset",
                snapshot_hash=prepared.snapshot_hash,
                split_version=prepared.split_version,
                config=config,
                inputs={
                    "ground_truth": str(
                        layout.ground_truth(prepared.snapshot_hash, prepared.gt_version)
                    ),
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
        "--prepared", default=None, help="Prepared-data id printed by prepare"
    )
    parser.add_argument("--role", action="append", dest="roles", help="Repeatable")
    parser.add_argument(
        "--fold", action="append", dest="folds", type=int, help="Repeatable"
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--run-id", default=None, help="Pipeline run id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for out_dir in build_datasets(
        load_config(args.config_path, require_api_key=False),
        args.prepared,
        args.roles,
        args.folds,
        args.processes,
        args.run_id,
    ):
        print(f"Dataset: {out_dir}")


if __name__ == "__main__":
    main()
