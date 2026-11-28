"""Evaluate predictions against ground truth (ADR 0005).

Ground truth and split come from ``datasets/{--dataset}`` (default: the run
id); predictions are read from, and results written to, the run ``--run-id``
names in :class:`bird_cv.core.layout.Layout`.

Modes:

- ``sweep``: val end-to-end macro-F1 for every ``(gap_fill_frames,
  det_conf)`` in ``[evaluate]``, from :meth:`Layout.sweep_infer_run`;
  writes ``sweep.parquet`` and ``best_setting.json``.
- ``test``: frozen-split test cameras, from :meth:`Layout.test_infer_run`.
- ``model``: one role's model-level metrics (e.g. window macro-F1 for a
  clip classifier, presence F1 for a detector) on the val and test splits
  of its frozen-split dataset, for ``--role`` and the checkpoint of training
  run ``--run-id``.

The dataset must match (by key) what the scored training or inference runs
recorded in their manifests.

Usage::

    python bird_cv/pipelines/evaluate.py CONFIG {sweep,test,model} \\
        --run-id ID [--dataset ID] [--role ROLE]
"""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl

from bird_cv.core.layout import EvalRun, Layout
from bird_cv.core.manifest import new_manifest, read_manifest_file, write_manifest
from bird_cv.evaluation.e2e import (
    evaluate_e2e,
    load_ground_truth,
    load_predictions,
    run_sweep,
    write_e2e_result,
)
from bird_cv.pipelines.config import BirdCVConfig, load_config
from bird_cv.pipelines.prepare import dataset_key
from bird_cv.training.registry import TrainContext, get_trainer

logger = logging.getLogger(__name__)


def check_dataset(
    dataset: dict[str, Any], manifests: Iterable[Path], what: str
) -> None:
    """Raise unless every manifest that records a dataset recorded ``dataset``.

    Datasets compare by key, so a reused (symlinked) dataset id matches the
    one it links to. Manifests that record none (inference with configured,
    untrained checkpoints) are logged, not rejected.

    Args:
        dataset: ``{"id", "key"}`` of the dataset this evaluation scores against.
        manifests: Manifest files of the runs being scored.
        what: The runs, for messages.

    Raises:
        FileNotFoundError: If there are no manifests.
        ValueError: If a manifest recorded a different dataset.
    """
    paths = list(manifests)
    if not paths:
        raise FileNotFoundError(f"No manifests for {what}")
    for path in paths:
        recorded = read_manifest_file(path).config.get("dataset")
        if recorded is None:
            logger.warning("%s records no dataset; can't cross-check", path)
        elif recorded["key"] != dataset["key"]:
            raise ValueError(
                f"{what} used dataset {recorded['id']} ({path}), but this "
                f"evaluation uses {dataset['id']}, which differs"
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
    dataset_id: str | None = None,
) -> Path:
    """Run one evaluation mode and write its outputs and manifest.

    Args:
        cfg: Pipeline config; uses ``[evaluate]`` (and ``[models]`` for
            ``model``).
        mode: ``"sweep"``, ``"test"`` or ``"model"``.
        run_id: The pipeline run to score: its inference runs, or for
            ``model`` its training run of ``role``. Also names the output.
        role: A ``[models]`` role, for ``model``.
        dataset_id: The dataset to score against; ``run_id`` if ``None``.

    Returns:
        The output directory.
    """
    ev = cfg.evaluate
    # No detector (cage-input classifier, ADR 0009): no boxes to score
    with_detection = "detector" in cfg.models.roles()
    layout = Layout(cfg.path_to_base_output)
    if mode not in ("sweep", "test", "model"):
        raise ValueError(f"Unknown mode {mode!r}")
    if not run_id:
        raise ValueError(f"{mode} mode needs the scored run's --run-id")
    if mode == "model" and role not in cfg.models.roles():
        raise ValueError(f"role must be one of {sorted(cfg.models.roles())}")
    dataset_id = layout.resolve_dataset(dataset_id, run_id)
    key = dataset_key(layout, dataset_id)
    dataset = {"id": dataset_id, "key": key}
    out = layout.evaluation_run(mode, run_id, role)
    out.mkdir(parents=True, exist_ok=True)
    results = EvalRun(out)
    print(f"Run directory: {out}")
    inputs: dict[str, str] = {}
    split = pl.read_parquet(layout.dataset_split_file(dataset_id))

    if mode == "model":
        assert role is not None  # checked above
        spec = cfg.models.roles()[role]
        train_dir = layout.train_run(role, run_id)
        check_dataset(
            dataset, [train_dir / "manifest.json"], f"training run {train_dir}"
        )
        checkpoint = get_trainer(spec.kind).checkpoint_path(train_dir)
        data_dir = layout.dataset(dataset_id, role, spec.dataset_name)
        context = TrainContext(
            device=cfg.train.device,
            select_conf=cfg.infer.det_conf,
            iou_threshold=ev.iou_threshold,
        )
        scores = {
            split_name: get_trainer(spec.kind).evaluate(
                checkpoint,
                spec,
                data_dir,
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
        inputs = {"dataset": str(data_dir), "checkpoint": str(checkpoint)}
    else:
        truth_path = layout.ground_truth(dataset_id)
        truth = load_ground_truth(truth_path)
        inputs["ground_truth"] = str(truth_path)

        if mode == "sweep":
            val_truth = _truth_for(truth, _cameras(split, "val"))
            runs = {
                setting: layout.sweep_infer_run(run_id, *setting)
                for setting in cfg.sweep_settings()
            }
            missing = [
                str(r.dir) for r in runs.values() if not r.predictions_dir.exists()
            ]
            if missing:
                raise FileNotFoundError(f"Missing sweep settings: {missing}")
            predictions = {}
            for setting, infer_run in runs.items():
                check_dataset(
                    dataset, infer_run.task_manifests(), f"inference {infer_run.dir}"
                )
                predictions[setting] = load_predictions(infer_run.predictions_dir)
            sweep_root = layout.sweep_root(run_id)
            sweep = run_sweep(
                val_truth,
                predictions,
                cfg.behaviors.classes,
                ev.iou_threshold,
                with_detection=with_detection,
            )
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
            check_dataset(
                dataset, test_run.task_manifests(), f"inference {test_run.dir}"
            )
            pred_dir = test_run.predictions_dir
            result = evaluate_e2e(
                _truth_for(truth, _cameras(split, "test")),
                load_predictions(pred_dir),
                cfg.behaviors.classes,
                ev.iou_threshold,
                with_detection=with_detection,
            )
            print(json.dumps(result.summary(), indent=2, default=float))
            print(result.per_camera)
            write_e2e_result(result, out)
            inputs["predictions"] = str(pred_dir)

    write_manifest(
        new_manifest(
            kind="evaluate",
            snapshot_hash=key["snapshot_hash"],
            split_version=key["split_version"],
            config={
                "mode": mode,
                "evaluate": asdict(ev),
                # One setting, the [infer] values, when no detector runs
                "sweep_settings": cfg.sweep_settings(),
                "dataset": dataset,
            },
            inputs=inputs,
        ),
        out,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument("mode", choices=("sweep", "test", "model"))
    parser.add_argument(
        "--run-id", default=None, help="Pipeline run to score (and output name)"
    )
    parser.add_argument("--role", default=None, help="[models] role, for model")
    parser.add_argument(
        "--dataset", default=None, help="Dataset id (default: --run-id)"
    )
    args = parser.parse_args()

    out = run_evaluate(
        load_config(args.config_path, require_api_key=False),
        args.mode,
        args.run_id,
        args.role,
        args.dataset,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
