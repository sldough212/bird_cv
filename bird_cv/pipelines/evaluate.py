"""Evaluate predictions against ground truth (ADR 0005).

Modes:

- ``sweep``: val end-to-end macro-F1 for every ``(gap_fill_frames,
  det_conf)`` in ``[evaluate]``, reading predictions from
  ``{predictions}/val_sweep/gap{g}_conf{c}/``; writes ``sweep.parquet`` and
  ``best_setting.json``.
- ``test``: frozen-split test cameras, predictions from ``{predictions}/test/``.
- ``cv``: 3-fold camera CV, predictions from ``{predictions}/cv/fold{k}/``;
  pooled, per-camera and per-fold results.
- ``windows``: VideoMAE window macro-F1 on the val and test clips of
  ``[train] mae_clips`` for ``--mae-model``.

Usage::

    python bird_cv/pipelines/evaluate.py CONFIG {sweep,test,cv,windows} \\
        [--run-id ID] [--mae-model DIR]
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.evaluation.cv import evaluate_cv
from bird_cv.evaluation.e2e import (
    evaluate_e2e,
    load_ground_truth,
    load_predictions,
    run_sweep,
    sweep_dir_name,
    sweep_settings,
    write_e2e_result,
)
from bird_cv.pipelines.config import BirdCVConfig, load_config, resolve_run_dir

MODES = ("sweep", "test", "cv", "windows")


def latest_split(splits_dir: Path) -> tuple[pl.DataFrame, int]:
    """The highest-numbered ``split_v{n}.parquet`` in ``splits_dir`` and its ``n``."""
    versions = {
        int(p.stem.removeprefix("split_v")): p
        for p in splits_dir.glob("split_v*.parquet")
    }
    if not versions:
        raise FileNotFoundError(f"No split_v{{n}}.parquet in {splits_dir}")
    n = max(versions)
    return pl.read_parquet(versions[n]), n


def _cameras(split: pl.DataFrame, name: str) -> pl.Series:
    return split.filter(pl.col("split") == name)["camera_id"]


def _truth_for(truth: pl.DataFrame, cameras: pl.Series) -> pl.DataFrame:
    return truth.filter(pl.col("camera_id").is_in(cameras.implode()))


def run_evaluate(
    cfg: BirdCVConfig,
    mode: str,
    run_id: str | None = None,
    mae_model: Path | None = None,
) -> Path:
    """Run one evaluation mode and write its outputs and manifest.

    Args:
        cfg: Pipeline config; uses ``[evaluate]`` (and ``[train]`` for
            ``windows``).
        mode: One of :data:`MODES`.
        run_id: Fixed output directory name; timestamped if ``None``.
        mae_model: VideoMAE checkpoint directory, for ``windows``.

    Returns:
        The output directory.
    """
    ev = cfg.evaluate
    base = cfg.path_to_base_output
    predictions_root = base / ev.predictions
    out = resolve_run_dir(base / ev.output / mode, run_id)
    inputs: dict[str, str] = {}
    split_version = None

    if mode == "windows":
        from bird_cv.classification.evaluate_video_model import evaluate_video_model

        if mae_model is None:
            raise ValueError("windows mode needs --mae-model")
        clips = base / cfg.train.mae_clips
        scores = {}
        for split_name in ("val", "test"):
            window_result = evaluate_video_model(
                clips_root=clips,
                model_path=mae_model,
                output_path=out / split_name,
                num_frames=cfg.train.mae_num_frames,
                split=split_name,
                window_macro_f1=True,
            )
            scores[split_name] = window_result["macro_f1"]
        (out / "window_macro_f1.json").write_text(json.dumps(scores, indent=2))
        inputs = {"clips": str(clips), "model": str(mae_model)}
    else:
        truth_path = base / ev.ground_truth
        truth = load_ground_truth(truth_path)
        inputs["ground_truth"] = str(truth_path)

        if mode == "sweep":
            split, split_version = latest_split(base / ev.splits)
            val_truth = _truth_for(truth, _cameras(split, "val"))
            predictions = {}
            for gap, conf in sweep_settings(
                ev.sweep_gap_fill_frames, ev.sweep_det_conf
            ):
                pred_dir = predictions_root / "val_sweep" / sweep_dir_name(gap, conf)
                if not pred_dir.exists():
                    print(f"Skipping missing sweep setting: {pred_dir}")
                    continue
                predictions[(gap, conf)] = load_predictions(pred_dir)
            if not predictions:
                raise FileNotFoundError(
                    f"No sweep predictions under {predictions_root}"
                )
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
            inputs["predictions"] = str(predictions_root / "val_sweep")

        elif mode == "test":
            split, split_version = latest_split(base / ev.splits)
            pred_dir = predictions_root / "test"
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
            folds_path = base / ev.cv_folds
            cv_folds = pl.read_parquet(folds_path)
            fold_predictions = {
                fold: load_predictions(predictions_root / "cv" / f"fold{fold}")
                for fold in sorted(cv_folds["fold"].unique().to_list())
            }
            cv = evaluate_cv(truth, cv_folds, fold_predictions, ev.iou_threshold)
            print(json.dumps(cv.pooled.summary(), indent=2, default=float))
            print(cv.pooled.per_camera)
            print(cv.per_fold)
            write_e2e_result(cv.pooled, out)
            cv.per_fold.write_parquet(out / "per_fold.parquet")
            inputs |= {
                "cv_folds": str(folds_path),
                "predictions": str(predictions_root / "cv"),
            }
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    write_manifest(
        new_manifest(
            kind="evaluate",
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
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--run-id", default=None, help="Output directory name")
    parser.add_argument("--mae-model", type=Path, default=None)
    args = parser.parse_args()

    out = run_evaluate(
        load_config(args.config_path), args.mode, args.run_id, args.mae_model
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
