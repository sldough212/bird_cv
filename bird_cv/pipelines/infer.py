"""Entry point: end-to-end inference on raw videos (no Label Studio involved).

Models are the ones the config picks: ``[infer.segmenter]``, and the
``[models.detector]`` / ``[models.classifier]`` kinds with checkpoints from
``--train-run-id`` (that run's outputs) or each table's ``checkpoint``.

``--split val --sweep-index K`` runs the frozen split's val cameras at the
K-th ``[evaluate]`` (gap fill, det_conf) setting into
:meth:`Layout.sweep_infer_run`; ``--split test`` runs the test cameras at the
setting ``evaluate sweep`` chose into :meth:`Layout.test_infer_run`. Otherwise
outputs go to :meth:`Layout.infer_run`.

Example::

    python bird_cv/pipelines/infer.py bird_cv/pipelines/configs/config.toml \\
        --camera H7,I22 --limit 4 --run-id test --task-index 0 --num-tasks 2
"""

import argparse
import dataclasses
import json
from pathlib import Path

import polars as pl

from bird_cv.core.layout import EvalRun, Layout
from bird_cv.evaluation.e2e import sweep_settings
from bird_cv.inference.pipeline import find_raw_videos, run_infer
from bird_cv.inference.registry import inference_models
from bird_cv.pipelines.config import default_run_id, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end inference.")
    parser.add_argument("config_path", type=Path, help="Path to the TOML config")
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Only this camera (repeatable; ids contain commas, e.g. H7,I22)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N videos (sorted)"
    )
    parser.add_argument(
        "--run-id", default=None, help="Run directory name (default: timestamp)"
    )
    parser.add_argument(
        "--train-run-id",
        default=None,
        help="Use this training run's checkpoints (default: [models.{role}] checkpoint)",
    )
    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--sweep-index", type=int, default=None)
    parser.add_argument("--prepared", default=None, help="Prepared-data id")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    models = inference_models(cfg, train_run_id=args.train_run_id)
    layout = Layout(cfg.path_to_base_output)
    run_id = default_run_id(args.run_id)
    infer_cfg = cfg.infer
    cameras = args.cameras
    run_dir = layout.infer_run(run_id).dir
    if args.split:
        if args.run_id is None or args.cameras:
            parser.error("--split needs --run-id and no --camera")
        prepared = layout.resolve_prepared(args.prepared, run_id)
        split = pl.read_parquet(layout.split(prepared.split_version))
        cameras = split.filter(pl.col("split") == args.split)["camera_id"].to_list()
        if args.split == "val":
            if args.sweep_index is None:
                parser.error("--split val needs --sweep-index")
            ev = cfg.evaluate
            gap, conf = sweep_settings(ev.sweep_gap_fill_frames, ev.sweep_det_conf)[
                args.sweep_index
            ]
            run_dir = layout.sweep_infer_run(run_id, gap, conf).dir
        else:
            best = EvalRun(layout.evaluation_run("sweep", run_id)).best_setting
            setting = json.loads(best.read_text())
            gap, conf = setting["gap_fill_frames"], setting["det_conf"]
            run_dir = layout.test_infer_run(run_id).dir
        infer_cfg = dataclasses.replace(infer_cfg, gap_fill_frames=gap, det_conf=conf)

    videos = find_raw_videos(cfg.path_to_raw_videos, cameras)[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found under {cfg.path_to_raw_videos}")
    print(f"Run directory: {run_dir}")

    failed = run_infer(
        videos=videos,
        run_dir=run_dir,
        models=models,
        cfg=infer_cfg,
        task_index=args.task_index,
        num_tasks=args.num_tasks,
        limit=args.limit,
    )
    if failed:
        raise SystemExit(
            f"{len(failed)} video(s) failed; see {run_dir}/failed_task*.json"
        )


if __name__ == "__main__":
    main()
