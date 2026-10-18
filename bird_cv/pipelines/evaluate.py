"""Evaluate predictions against ground truth (ADR 0005).

Ground truth, split and CV folds are the current prepared data
(``prepared/current.json``); predictions are read from, and results written
to, the run ``--run-id`` names in :class:`bird_cv.core.layout.Layout`.

Modes:

- ``sweep``: val end-to-end macro-F1 for every ``(gap_fill_frames,
  det_conf)`` in ``[evaluate]``, from :meth:`Layout.sweep_infer_run`;
  writes ``sweep.parquet`` and ``best_setting.json``.
- ``test``: frozen-split test cameras, from :meth:`Layout.test_infer_run`.
- ``cv``: 3-fold camera CV, from :meth:`Layout.cv_infer_run`; pooled,
  per-camera and per-fold results.
- ``model``: one role's model-level metrics (e.g. window macro-F1 for a
  clip classifier, presence F1 for a detector) on the val and test splits
  of its frozen-split dataset, for ``--role`` and the checkpoint of training
  run ``--run-id``.

``--prepared`` is the id prepare prints last; without it the last prepare
output is used, with a warning. The prepared data must match what the
scored training or inference runs recorded in their manifests.

Usage::

    python bird_cv/pipelines/evaluate.py CONFIG {sweep,test,cv,model} \\
        --run-id ID --prepared ID [--role ROLE]
"""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import polars as pl

from bird_cv.core.layout import EvalRun, Layout, Prepared
from bird_cv.core.manifest import new_manifest, read_manifest_file, write_manifest
from bird_cv.evaluation.cv import evaluate_cv
from bird_cv.evaluation.e2e import (
    evaluate_e2e,
    load_ground_truth,
    load_predictions,
    run_sweep,
    sweep_settings,
    write_e2e_result,
)
from bird_cv.pipelines.config import BirdCVConfig, load_config

logger = logging.getLogger(__name__)


def check_prepared(prepared: Prepared, manifests: Iterable[Path], what: str) -> None:
    """Raise unless every manifest that records prepared data recorded ``prepared``.

    Manifests that record none (inference with configured, untrained
    checkpoints) are logged, not rejected.

    Args:
        prepared: The prepared data this evaluation scores against.
        manifests: Manifest files of the runs being scored.
        what: The runs, for messages.

    Raises:
        FileNotFoundError: If there are no manifests.
        ValueError: If a manifest recorded different prepared data.
    """
    paths = list(manifests)
    if not paths:
        raise FileNotFoundError(f"No manifests for {what}")
    expected = asdict(prepared)
    for path in paths:
        recorded = read_manifest_file(path).config.get("prepared")
        if recorded is None:
            logger.warning("%s records no prepared data; can't cross-check", path)
        elif recorded != expected:
            raise ValueError(
                f"{what} used prepared data {Prepared(**recorded).id} ({path}), "
                f"but this evaluation uses {prepared.id}"
            )


def _cameras(split: pl.DataFrame, name: str) -> pl.Series:
    return split.filter(pl.col("split") == name)["camera_id"]


def _truth_for(truth: pl.DataFrame, cameras: pl.Series) -> pl.DataFrame:
    return truth.filter(pl.col("camera_id").is_in(cameras.implode()))


def run_evaluate(
    cfg: BirdCVConfig,
    mode: str,
    run_id: str | None = None,
    role: str | None = None,
    prepared_id: str | None = None,
) -> Path:
    """Run one evaluation mode and write its outputs and manifest.

    Args:
        cfg: Pipeline config; uses ``[evaluate]`` (and ``[models]`` for
            ``model``).
        mode: ``"sweep"``, ``"test"``, ``"cv"`` or ``"model"``.
        run_id: The pipeline run to score: its inference runs, or for
            ``model`` its training run of ``role``. Also names the output.
        role: A ``[models]`` role, for ``model``.
        prepared_id: :attr:`Prepared.id` of the data to score against; the
            last prepare output (logged) if ``None``.

    Returns:
        The output directory.
    """
    ev = cfg.evaluate
    layout = Layout(cfg.path_to_base_output)
    if mode not in ("sweep", "test", "cv", "model"):
        raise ValueError(f"Unknown mode {mode!r}")
    if not run_id:
        raise ValueError(f"{mode} mode needs the scored run's --run-id")
    if mode == "model" and role not in cfg.models.roles():
        raise ValueError(f"role must be one of {sorted(cfg.models.roles())}")
    prepared = layout.resolve_prepared(prepared_id, run_id)
    out = layout.evaluation_run(mode, run_id, role)
    out.mkdir(parents=True, exist_ok=True)
    results = EvalRun(out)
    print(f"Run directory: {out}")
    inputs: dict[str, str] = {}
    split_version = prepared.split_version
    split = pl.read_parquet(layout.split(split_version))

    if mode == "model":
        from bird_cv.training.registry import TrainContext, get_trainer

        assert role is not None  # checked above
        spec = cfg.models.roles()[role]
        train_dir = layout.train_run(role, run_id)
        check_prepared(
            prepared, [train_dir / "manifest.json"], f"training run {train_dir}"
        )
        checkpoint = get_trainer(spec.kind).checkpoint_path(train_dir)
        dataset = layout.dataset(prepared, role, spec.kind)
        context = TrainContext(
            device=cfg.train.device,
            select_conf=cfg.infer.det_conf,
            iou_threshold=ev.iou_threshold,
        )
        scores = {
            split_name: get_trainer(spec.kind).evaluate(
                checkpoint,
                spec,
                dataset,
                split_name,
                results.model_split(split_name),
                context,
            )
            for split_name in ("val", "test")
        }
        print(json.dumps(scores, indent=2, default=float))
        results.model_metrics.write_text(
            json.dumps(
                {"role": role, "kind": spec.kind} | scores, indent=2, default=float
            )
        )
        inputs = {"dataset": str(dataset), "checkpoint": str(checkpoint)}
    else:
        truth_path = layout.ground_truth(prepared.snapshot_hash, prepared.gt_version)
        truth = load_ground_truth(truth_path)
        inputs["ground_truth"] = str(truth_path)

        if mode == "sweep":
            val_truth = _truth_for(truth, _cameras(split, "val"))
            predictions = {}
            for gap, conf in sweep_settings(
                ev.sweep_gap_fill_frames, ev.sweep_det_conf
            ):
                infer_run = layout.sweep_infer_run(run_id, gap, conf)
                if not infer_run.predictions_dir.exists():
                    print(f"Skipping missing sweep setting: {infer_run.dir}")
                    continue
                check_prepared(
                    prepared, infer_run.task_manifests(), f"inference {infer_run.dir}"
                )
                predictions[(gap, conf)] = load_predictions(infer_run.predictions_dir)
            sweep_root = layout.sweep_root(run_id)
            if not predictions:
                raise FileNotFoundError(f"No sweep predictions under {sweep_root}")
            sweep = run_sweep(val_truth, predictions, ev.iou_threshold)
            print(sweep)
            sweep.write_parquet(results.sweep)
            best = sweep.row(0, named=True)
            results.best_setting.write_text(
                json.dumps(
                    {k: best[k] for k in ("gap_fill_frames", "det_conf", "macro_f1")},
                    indent=2,
                )
            )
            inputs["predictions"] = str(sweep_root)

        elif mode == "test":
            test_run = layout.test_infer_run(run_id)
            check_prepared(
                prepared, test_run.task_manifests(), f"inference {test_run.dir}"
            )
            pred_dir = test_run.predictions_dir
            result = evaluate_e2e(
                _truth_for(truth, _cameras(split, "test")),
                load_predictions(pred_dir),
                ev.iou_threshold,
            )
            print(json.dumps(result.summary(), indent=2, default=float))
            print(result.per_camera)
            write_e2e_result(result, out)
            inputs["predictions"] = str(pred_dir)

        elif mode == "cv":
            folds_path = layout.cv_folds(split_version)
            cv_folds = pl.read_parquet(folds_path)
            fold_runs = {
                fold: layout.cv_infer_run(run_id, fold)
                for fold in sorted(cv_folds["fold"].unique().to_list())
            }
            for fold_run in fold_runs.values():
                check_prepared(
                    prepared, fold_run.task_manifests(), f"inference {fold_run.dir}"
                )
            fold_predictions = {
                fold: load_predictions(fold_run.predictions_dir)
                for fold, fold_run in fold_runs.items()
            }
            cv = evaluate_cv(truth, cv_folds, fold_predictions, ev.iou_threshold)
            print(json.dumps(cv.pooled.summary(), indent=2, default=float))
            print(cv.pooled.per_camera)
            print(cv.per_fold)
            write_e2e_result(cv.pooled, out)
            cv.per_fold.write_parquet(results.per_fold)
            # Hyperparameters and the gap-fill/det_conf setting were chosen on
            # the frozen split's val cameras, which are also CV test cameras.
            results.cv_summary.write_text(
                json.dumps(
                    {
                        "pooled": cv.pooled.summary(),
                        "tuning_and_sweep_val_cameras": sorted(
                            _cameras(split, "val").to_list()
                        ),
                        "tuning_and_sweep_split_version": split_version,
                        "note": (
                            "Model hyperparameters and the gap-fill/det_conf "
                            "sweep were selected on the frozen split's val "
                            "cameras; every fold reuses them."
                        ),
                    },
                    indent=2,
                    default=float,
                )
            )
            inputs |= {
                "cv_folds": str(folds_path),
                "predictions": str(layout.cv_root(run_id)),
            }

    write_manifest(
        new_manifest(
            kind="evaluate",
            snapshot_hash=prepared.snapshot_hash,
            split_version=split_version,
            config={
                "mode": mode,
                "evaluate": asdict(ev),
                "prepared": asdict(prepared),
            },
            inputs=inputs,
        ),
        out,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument("mode", choices=("sweep", "test", "cv", "model"))
    parser.add_argument(
        "--run-id", default=None, help="Pipeline run to score (and output name)"
    )
    parser.add_argument("--role", default=None, help="[models] role, for model")
    parser.add_argument(
        "--prepared", default=None, help="Prepared-data id printed by prepare"
    )
    args = parser.parse_args()

    out = run_evaluate(
        load_config(args.config_path, require_api_key=False),
        args.mode,
        args.run_id,
        args.role,
        args.prepared,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
