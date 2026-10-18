"""Train a pipeline role's model from generic pretrained weights (ADR 0004).

Roles are the :class:`~bird_cv.pipelines.config.ModelSpec` tables of
``[models]`` (``detector``, ``classifier``); each names a ``kind`` that
:func:`bird_cv.training.registry.get_trainer` turns into a trainer. On the
frozen split a role is tuned (if ``tune``) and selected on its validation
objective (ADR 0005). With ``--fold k`` it is retrained on CV fold ``k``'s
dataset with fixed hyperparameters from a frozen-split run (ADR 0001), no
tuning. ``--smoke`` shrinks tuning and epochs to 2 for a first cluster run
(a smoke test; never report its scores). The ``paths`` mode prints each
role's resolved run directory and checkpoint for a run id as shell
assignments, without creating anything.

Usage::

    python bird_cv/pipelines/train.py CONFIG ROLE --prepared ID [--run-id ID] [--smoke]
    python bird_cv/pipelines/train.py CONFIG ROLE --prepared ID --fold K \\
        --hyperparameters RUN_DIR/best_hyperparameters.json [--run-id ID]
    python bird_cv/pipelines/train.py CONFIG paths --run-id ID

``--prepared`` is the id prepare prints last (:attr:`Prepared.id`); without
it the last prepare output is used, with a warning.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from bird_cv.core.layout import Layout, TrainRun
from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.evaluation.cv import check_fold_cameras
from bird_cv.pipelines.config import BirdCVConfig, default_run_id, load_config

if TYPE_CHECKING:
    from bird_cv.training.registry import Trainer


def smoke_config(cfg: BirdCVConfig) -> BirdCVConfig:
    """Copy of ``cfg`` with 2 tuning trials and 2 epochs for every role.

    Options are left alone; both tuners already cap their ASHA grace period
    at the epoch count.
    """
    smoke: dict[str, Any] = {
        role: dataclasses.replace(spec, tune_trials=2, epochs=2)
        for role, spec in cfg.models.roles().items()
    }
    return dataclasses.replace(cfg, models=dataclasses.replace(cfg.models, **smoke))


def check_fold(
    fold: int | None,
    hyperparameters: dict | None,
    dataset: Path,
    trainer: "Trainer",
    *,
    split_path: Path,
    cv_folds_path: Path,
) -> None:
    """Folds reuse frozen-split hyperparameters and never train on test cameras.

    A fold dataset's train/val cameras are the union of those listed in
    ``split_path`` (the :func:`bird_cv.evaluation.cv.fold_split` output it
    was built from) and those its layout shows
    (:meth:`Trainer.dataset_cameras`).

    Raises:
        ValueError: If a fold has no hyperparameters, no cameras can be
            found, or any found camera is one of the fold's test cameras.
    """
    if fold is None:
        return
    if hyperparameters is None:
        raise ValueError(
            "CV folds reuse the frozen-split hyperparameters; pass --hyperparameters"
        )
    cameras = trainer.dataset_cameras(dataset)
    if split_path.exists():
        split = pl.read_parquet(split_path)
        cameras |= set(
            split.filter(pl.col("split").is_in(["train", "val"]))["camera_id"]
        )
    if not cameras:
        raise ValueError(
            f"Can't tell which cameras {dataset} trains on: write its "
            f"fold_split output to {split_path}"
        )
    check_fold_cameras(pl.read_parquet(cv_folds_path), fold, cameras)


def run_train(
    cfg: BirdCVConfig,
    role: str,
    run_id: str | None = None,
    fold: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
    prepared_id: str | None = None,
) -> Path:
    """Tune (optionally) and train one role's model; write results and a manifest.

    Trains on :meth:`Layout.dataset` of the prepared data and writes the
    :class:`TrainRun` results and ``manifest.json`` (which records the
    prepared data) to :meth:`Layout.train_run`; the checkpoint goes where
    the role's trainer puts it (:meth:`Trainer.checkpoint_path`).

    Args:
        cfg: Pipeline config; uses ``[models]``, ``[train]``, ``infer.det_conf`` and
            ``evaluate.iou_threshold`` (detector selection).
        role: A role from ``ModelsConfig.roles()``.
        run_id: Fixed run directory name; timestamped if ``None``.
        fold: CV fold to train on; ``None`` for the frozen split.
        hyperparameters: Fixed hyperparameters (a frozen-split run's
            ``best_hyperparameters.json``). Disables tuning.
        prepared_id: :attr:`Prepared.id` of the data to train on; the last
            prepare output (logged) if ``None``.

    Returns:
        The run directory.
    """
    from bird_cv.training.registry import TrainContext, get_trainer

    spec = cfg.models.roles()[role]
    trainer = get_trainer(spec.kind)
    layout = Layout(cfg.path_to_base_output)
    prepared = layout.resolve_prepared(prepared_id, run_id)
    dataset = layout.dataset(prepared, role, spec.kind, fold)
    check_fold(
        fold,
        hyperparameters,
        dataset,
        trainer,
        split_path=layout.dataset_split(prepared, role, spec.kind, fold),
        cv_folds_path=layout.cv_folds(prepared.split_version),
    )
    run_dir = layout.train_run(role, default_run_id(run_id), fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    context = TrainContext(
        device=cfg.train.device,
        select_conf=cfg.infer.det_conf,
        iou_threshold=cfg.evaluate.iou_threshold,
    )

    if hyperparameters is None and spec.tune:
        result = trainer.tune(spec, dataset, run_dir, context)
    else:
        result = trainer.train(spec, dataset, run_dir, hyperparameters or {}, context)

    run = TrainRun(run_dir)
    run.best_hyperparameters.write_text(
        json.dumps(result.hyperparameters, indent=2, default=float)
    )
    run.train_result.write_text(
        json.dumps(
            {"checkpoint": str(result.checkpoint), "metrics": result.metrics},
            indent=2,
            default=float,
        )
    )
    write_manifest(
        new_manifest(
            kind=f"train_{role}",
            snapshot_hash=prepared.snapshot_hash,
            split_version=prepared.split_version,
            config={
                "role": role,
                "spec": dataclasses.asdict(spec),
                "fold": fold,
                "prepared": dataclasses.asdict(prepared),
            },
            inputs={
                "dataset": str(dataset),
                "pretrained": spec.pretrained,
                "checkpoint": str(result.checkpoint),
            },
        ),
        run_dir,
    )
    return run_dir


def print_paths(cfg: BirdCVConfig, run_id: str) -> None:
    """Print a pipeline run's paths and sizes as shell assignments.

    Variables are ``ROLES`` (space-separated role names), ``BASE``,
    ``PREPARED_FILE``, ``N_SWEEP`` (val-sweep settings), ``INFER_DIR``,
    ``SWEEP_DIR`` (sweep evaluation), ``TEST_DIR`` (test evaluation),
    ``{ROLE}_RUN_DIR`` and ``{ROLE}_BEST``.
    """
    from bird_cv.evaluation.e2e import sweep_settings
    from bird_cv.training.registry import get_trainer

    layout = Layout(cfg.path_to_base_output)
    ev = cfg.evaluate
    print(f"ROLES='{' '.join(cfg.models.roles())}'")
    print(f"BASE='{cfg.path_to_base_output}'")
    print(f"PREPARED_FILE='{layout.run_prepared(run_id)}'")
    print(f"N_SWEEP={len(sweep_settings(ev.sweep_gap_fill_frames, ev.sweep_det_conf))}")
    print(f"INFER_DIR='{layout.infer_run(run_id).dir}'")
    print(f"SWEEP_DIR='{layout.evaluation_run('sweep', run_id)}'")
    print(f"TEST_DIR='{layout.evaluation_run('test', run_id)}'")
    for role, spec in cfg.models.roles().items():
        run_dir = layout.train_run(role, run_id)
        best = get_trainer(spec.kind).checkpoint_path(run_dir)
        print(f"{role.upper()}_RUN_DIR='{run_dir}'")
        print(f"{role.upper()}_BEST='{best}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument("role", help="A [models] role (detector, classifier) or paths")
    parser.add_argument("--run-id", default=None, help="Run directory name")
    parser.add_argument(
        "--prepared", default=None, help="Prepared-data id printed by prepare"
    )
    parser.add_argument("--fold", type=int, default=None, help="CV fold to train on")
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="best_hyperparameters.json from a frozen-split run (disables tuning)",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="2 tuning trials and 2 epochs"
    )
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.role == "paths":
        if not args.run_id:
            parser.error("paths needs --run-id")
        print_paths(cfg, args.run_id)
        return
    if args.role not in cfg.models.roles():
        parser.error(f"role must be one of {sorted(cfg.models.roles())} or paths")

    hyperparameters = (
        json.loads(args.hyperparameters.read_text()) if args.hyperparameters else None
    )
    run_dir = run_train(
        cfg, args.role, args.run_id, args.fold, hyperparameters, args.prepared
    )
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
