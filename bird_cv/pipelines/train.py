"""Train a pipeline role's model from generic pretrained weights (ADR 0004).

Roles are the :class:`~bird_cv.pipelines.config.ModelSpec` tables of
``[train]`` (``detector``, ``classifier``); each names a ``kind`` that
:func:`bird_cv.training.registry.get_trainer` turns into a trainer. On the
frozen split a role is tuned (if ``tune``) and selected on its validation
objective (ADR 0005). With ``--fold k`` it is retrained on CV fold ``k``'s
dataset with fixed hyperparameters from a frozen-split run (ADR 0001), no
tuning. ``--smoke`` shrinks tuning and epochs to 2 for a first cluster run
(a smoke test; never report its scores). The ``paths`` mode prints each
role's resolved run directory and checkpoint for a run id as shell
assignments, without creating anything.

Usage::

    python bird_cv/pipelines/train.py CONFIG ROLE [--run-id ID] [--smoke]
    python bird_cv/pipelines/train.py CONFIG ROLE --fold K \\
        --hyperparameters RUN_DIR/best_hyperparameters.json [--run-id ID]
    python bird_cv/pipelines/train.py CONFIG paths --run-id ID
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.manifest import new_manifest, read_manifest, write_manifest
from bird_cv.evaluation.cv import check_fold_cameras
from bird_cv.pipelines.config import (
    BirdCVConfig,
    ModelSpec,
    load_config,
    resolve_run_dir,
)


def smoke_config(cfg: BirdCVConfig) -> BirdCVConfig:
    """Copy of ``cfg`` with 2 tuning trials and 2 epochs for every role."""
    smoke: dict[str, Any] = {
        role: dataclasses.replace(
            spec,
            tune_trials=2,
            epochs=2,
            options=spec.options | {"grace_period": 1},
        )
        for role, spec in cfg.train.roles().items()
    }
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **smoke))


def _dataset_dir(cfg: BirdCVConfig, spec: ModelSpec, fold: int | None) -> Path:
    root = cfg.path_to_base_output / spec.dataset
    return root / f"fold{fold}" if fold is not None else root


def _run_base(cfg: BirdCVConfig, role: str, fold: int | None) -> Path:
    base = cfg.path_to_base_output / cfg.train.output / role
    return base / f"fold{fold}" if fold is not None else base


def _check_fold(
    cfg: BirdCVConfig,
    fold: int | None,
    hyperparameters: dict | None,
    dataset: Path,
    split_file: str = "split.parquet",
) -> None:
    """Folds reuse frozen-split hyperparameters and never train on test cameras.

    The fold dataset's cameras come from ``{dataset}/{split_file}``: the
    :func:`bird_cv.evaluation.cv.fold_split` output it was built from.
    """
    if fold is None:
        return
    if hyperparameters is None:
        raise ValueError(
            "CV folds reuse the frozen-split hyperparameters; pass --hyperparameters"
        )
    path = dataset / split_file
    if not path.exists():
        raise ValueError(
            f"Can't tell which cameras {dataset} trains on: write its "
            f"fold_split output to {path}"
        )
    split = pl.read_parquet(path)
    cameras = set(split.filter(pl.col("split").is_in(["train", "val"]))["camera_id"])
    cv_folds = pl.read_parquet(cfg.path_to_base_output / cfg.evaluate.cv_folds)
    check_fold_cameras(cv_folds, fold, cameras)


def run_train(
    cfg: BirdCVConfig,
    role: str,
    run_id: str | None = None,
    fold: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
) -> Path:
    """Tune (optionally) and train one role's model; write results and a manifest.

    Writes ``best_hyperparameters.json``, ``train_result.json`` and
    ``manifest.json`` to the run directory; the checkpoint goes where the
    role's trainer puts it (:meth:`Trainer.checkpoint_path`).

    Args:
        cfg: Pipeline config; uses ``[train]``, ``infer.det_conf`` and
            ``evaluate.iou_threshold`` (detector selection) and
            ``evaluate.cv_folds`` (folds).
        role: A role from ``TrainConfig.roles()``.
        run_id: Fixed run directory name; timestamped if ``None``.
        fold: CV fold to train on; ``None`` for the frozen split.
        hyperparameters: Fixed hyperparameters (a frozen-split run's
            ``best_hyperparameters.json``). Disables tuning.

    Returns:
        The run directory.
    """
    from bird_cv.training.registry import TrainContext, get_trainer

    spec = cfg.train.roles()[role]
    trainer = get_trainer(spec.kind)
    dataset = _dataset_dir(cfg, spec, fold)
    _check_fold(cfg, fold, hyperparameters, dataset)
    run_dir = resolve_run_dir(_run_base(cfg, role, fold), run_id)
    context = TrainContext(
        device=cfg.train.device,
        select_conf=cfg.infer.det_conf,
        iou_threshold=cfg.evaluate.iou_threshold,
    )

    if hyperparameters is None and spec.tune:
        result = trainer.tune(spec, dataset, run_dir, context)
    else:
        result = trainer.train(spec, dataset, run_dir, hyperparameters or {}, context)

    (run_dir / "best_hyperparameters.json").write_text(
        json.dumps(result.hyperparameters, indent=2, default=float)
    )
    (run_dir / "train_result.json").write_text(
        json.dumps(
            {"checkpoint": str(result.checkpoint), "metrics": result.metrics},
            indent=2,
            default=float,
        )
    )
    upstream = read_manifest(dataset) if (dataset / "manifest.json").exists() else None
    write_manifest(
        new_manifest(
            kind=f"train_{role}",
            snapshot_hash=upstream.snapshot_hash if upstream else None,
            split_version=upstream.split_version if upstream else None,
            config={"role": role, "spec": dataclasses.asdict(spec), "fold": fold},
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
    """Print each role's frozen-split run dir and checkpoint as shell assignments.

    Variables are ``BASE``, ``{ROLE}_RUN_DIR`` and ``{ROLE}_BEST``.
    """
    from bird_cv.training.registry import get_trainer

    print(f"BASE='{cfg.path_to_base_output}'")
    for role, spec in cfg.train.roles().items():
        run_dir = _run_base(cfg, role, None) / run_id
        best = get_trainer(spec.kind).checkpoint_path(run_dir)
        print(f"{role.upper()}_RUN_DIR='{run_dir}'")
        print(f"{role.upper()}_BEST='{best}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument("role", help="A [train] role (detector, classifier) or paths")
    parser.add_argument("--run-id", default=None, help="Run directory name")
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
    if args.role not in cfg.train.roles():
        parser.error(f"role must be one of {sorted(cfg.train.roles())} or paths")

    hyperparameters = (
        json.loads(args.hyperparameters.read_text()) if args.hyperparameters else None
    )
    run_dir = run_train(cfg, args.role, args.run_id, args.fold, hyperparameters)
    print(f"Done: {run_dir}")


if __name__ == "__main__":
    main()
