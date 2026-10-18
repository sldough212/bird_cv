"""Entry point: end-to-end inference on raw videos (no Label Studio involved).

Models are the ones the config picks: ``[infer.segmenter]``, and the
``[train.detector]`` / ``[train.classifier]`` kinds with checkpoints from
``--train-run-id`` (that run's outputs) or each table's ``checkpoint``.
Outputs go to :meth:`bird_cv.core.layout.Layout.infer_run`.

Example::

    python bird_cv/pipelines/infer.py bird_cv/pipelines/configs/config.toml \\
        --camera H7,I22 --limit 4 --run-id test --task-index 0 --num-tasks 2
"""

import argparse
from pathlib import Path

from bird_cv.core.layout import Layout
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
        help="Use this training run's checkpoints (default: [train.{role}] checkpoint)",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    models = inference_models(cfg, train_run_id=args.train_run_id)

    videos = find_raw_videos(cfg.path_to_raw_videos, args.cameras)[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found under {cfg.path_to_raw_videos}")
    run_dir = Layout(cfg.path_to_base_output).infer_run(default_run_id(args.run_id)).dir
    print(f"Run directory: {run_dir}")

    failed = run_infer(
        videos=videos,
        run_dir=run_dir,
        models=models,
        cfg=cfg.infer,
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
