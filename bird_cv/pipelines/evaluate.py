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
  of its frozen-split dataset, for ``--role`` and ``--checkpoint``.

Usage::

    python bird_cv/pipelines/evaluate.py CONFIG {sweep,test,cv} --run-id ID
    python bird_cv/pipelines/evaluate.py CONFIG model --role ROLE \\
        --checkpoint PATH [--run-id ID]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from bird_cv.core.layout import Layout
from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.evaluation.cv import evaluate_cv
from bird_cv.evaluation.e2e import (
    evaluate_e2e,
    load_ground_truth,
    load_predictions,
    run_sweep,
    sweep_settings,
    write_e2e_result,
)
from bird_cv.pipelines.config import BirdCVConfig, default_run_id, load_config


def _cameras(split: pl.DataFrame, name: str) -> pl.Series:
    return split.filter(pl.col("split") == name)["camera_id"]


def _truth_for(truth: pl.DataFrame, cameras: pl.Series) -> pl.DataFrame:
    return truth.filter(pl.col("camera_id").is_in(cameras.implode()))


def run_evaluate(
    cfg: BirdCVConfig,
    mode: str,
    run_id: str | None = None,
    role: str | None = None,
    checkpoint: Path | None = None,
) -> Path:
    """Run one evaluation mode and write its outputs and manifest.

    Args:
        cfg: Pipeline config; uses ``[evaluate]`` (and ``[train]`` for
            ``model``).
        mode: ``"sweep"``, ``"test"``, ``"cv"`` or ``"model"``.
        run_id: The pipeline run whose inference outputs to score (required
            except for ``model``, where ``None`` means a timestamp); also
            names the output directory.
        role: A ``[train]`` role, for ``model``.
        checkpoint: That role's checkpoint, for ``model``.

    Returns:
        The output directory.
    """
    ev = cfg.evaluate
    layout = Layout(cfg.path_to_base_output)
    prepared = layout.read_current()
    if mode == "model":
        if role not in cfg.train.roles():
            raise ValueError(f"role must be one of {sorted(cfg.train.roles())}")
        if checkpoint is None:
            raise ValueError("model mode needs --checkpoint")
    elif mode in ("sweep", "test", "cv"):
        if not run_id:
            raise ValueError(f"{mode} mode needs the inference run's --run-id")
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    out = layout.evaluation_run(mode, default_run_id(run_id), role)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out}")
    inputs: dict[str, str] = {}
    split_version = prepared.split_version
    split = pl.read_parquet(layout.split(split_version))

    if mode == "model":
        from bird_cv.training.registry import TrainContext, get_trainer

        assert role is not None and checkpoint is not None  # checked above
        spec = cfg.train.roles()[role]
        dataset = layout.dataset(prepared, role)
        context = TrainContext(
            device=cfg.train.device,
            select_conf=cfg.infer.det_conf,
            iou_threshold=ev.iou_threshold,
        )
        scores = {
            split_name: get_trainer(spec.kind).evaluate(
                checkpoint, spec, dataset, split_name, out / split_name, context
            )
            for split_name in ("val", "test")
        }
        print(json.dumps(scores, indent=2, default=float))
        (out / "model_metrics.json").write_text(
            json.dumps(
                {"role": role, "kind": spec.kind} | scores, indent=2, default=float
            )
        )
        inputs = {"dataset": str(dataset), "checkpoint": str(checkpoint)}
    else:
        assert run_id  # checked above
        truth_path = layout.ground_truth(prepared.snapshot_hash, prepared.gt_version)
        truth = load_ground_truth(truth_path)
        inputs["ground_truth"] = str(truth_path)

        if mode == "sweep":
            val_truth = _truth_for(truth, _cameras(split, "val"))
            predictions = {}
            for gap, conf in sweep_settings(
                ev.sweep_gap_fill_frames, ev.sweep_det_conf
            ):
                pred_dir = layout.sweep_infer_run(run_id, gap, conf).predictions_dir
                if not pred_dir.exists():
                    print(f"Skipping missing sweep setting: {pred_dir}")
                    continue
                predictions[(gap, conf)] = load_predictions(pred_dir)
            sweep_root = layout.sweep_root(run_id)
            if not predictions:
                raise FileNotFoundError(f"No sweep predictions under {sweep_root}")
            sweep = run_sweep(val_truth, predictions, ev.iou_threshold)
            print(sweep)
            sweep.write_parquet(out / "sweep.parquet")
            best = sweep.row(0, named=True)
            (out / "best_setting.json").write_text(
                json.dumps(
                    {k: best[k] for k in ("gap_fill_frames", "det_conf", "macro_f1")},
                    indent=2,
                )
            )
            inputs["predictions"] = str(sweep_root)

        elif mode == "test":
            pred_dir = layout.test_infer_run(run_id).predictions_dir
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
            fold_predictions = {
                fold: load_predictions(
                    layout.cv_infer_run(run_id, fold).predictions_dir
                )
                for fold in sorted(cv_folds["fold"].unique().to_list())
            }
            cv = evaluate_cv(truth, cv_folds, fold_predictions, ev.iou_threshold)
            print(json.dumps(cv.pooled.summary(), indent=2, default=float))
            print(cv.pooled.per_camera)
            print(cv.per_fold)
            write_e2e_result(cv.pooled, out)
            cv.per_fold.write_parquet(out / "per_fold.parquet")
            # Hyperparameters and the gap-fill/det_conf setting were chosen on
            # the frozen split's val cameras, which are also CV test cameras.
            (out / "cv.json").write_text(
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
            config={"mode": mode, "evaluate": asdict(ev)},
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
    parser.add_argument("--role", default=None, help="[train] role, for model")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    out = run_evaluate(
        load_config(args.config_path, require_api_key=False),
        args.mode,
        args.run_id,
        args.role,
        args.checkpoint,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
