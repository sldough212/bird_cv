"""Entry point: end-to-end inference on raw videos (no Label Studio involved).

Example::

    python bird_cv/pipelines/infer.py bird_cv/pipelines/configs/config.toml \\
        --camera H7,I22 --limit 4 --run-id test --task-index 0 --num-tasks 2
"""

import argparse
import os
from pathlib import Path

from bird_cv.inference.pipeline import find_raw_videos, run_infer
from bird_cv.pipelines.config import API_KEY_ENV_VAR, load_config, resolve_run_dir


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
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    # load_config requires a Label Studio key; inference never uses it
    os.environ.setdefault(API_KEY_ENV_VAR, "unused-by-infer")
    cfg = load_config(args.config_path)

    videos = find_raw_videos(cfg.path_to_raw_videos, args.cameras)[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found under {cfg.path_to_raw_videos}")
    run_dir = resolve_run_dir(cfg.path_to_base_output / "infer", args.run_id)

    run_infer(
        videos=videos,
        run_dir=run_dir,
        segmentation_configs_path=cfg.path_to_segmentation_configs,
        sam_model_id=cfg.sam_model_id,
        yolo_path=cfg.detect.path_to_yolo,
        tracker_path=cfg.detect.path_to_tracker_config,
        videomae_path=cfg.classify.path_to_mae,
        cfg=cfg.infer,
        task_index=args.task_index,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()
