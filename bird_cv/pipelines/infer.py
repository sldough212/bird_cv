"""Entry point: end-to-end inference on raw videos (no Label Studio involved).

Models are the ones the config picks: ``[infer.segmenter]``, and the
``[models.detector]`` / ``[models.classifier]`` kinds with checkpoints from
``--train-run-id`` (that run's outputs) or each table's ``checkpoint``.

``--split val --sweep-index K`` runs the frozen split's labeled val videos at
the K-th ``[evaluate]`` (gap fill, det_conf) setting into
:meth:`Layout.sweep_infer_run`; ``--split test`` runs the labeled test videos
at the setting ``evaluate sweep`` chose into :meth:`Layout.test_infer_run`.
Otherwise outputs go to :meth:`Layout.infer_run`.

Example::

    python bird_cv/pipelines/infer.py bird_cv/pipelines/configs/config.toml \\
        --camera H7,I22 --limit 4 --run-id test --task-index 0 --num-tasks 2
"""

import argparse
import dataclasses
import json
from pathlib import Path

import polars as pl

from bird_cv.core.ids import parse_raw_video_path
from bird_cv.core.layout import EvalRun, Layout, default_run_id
from bird_cv.inference.pipeline import run_infer
from bird_cv.inference.registry import inference_models
from bird_cv.preprocessing.standardize import find_raw_videos
from bird_cv.pipelines.config import BirdCVConfig, InferConfig, load_config
from bird_cv.pipelines.prepare import dataset_key


def split_run(
    cfg: BirdCVConfig,
    run_id: str,
    split: str,
    sweep_index: int | None,
    dataset_id: str | None,
) -> tuple[list[Path], Path, InferConfig, dict]:
    """Videos, run dir and settings of a ``--split`` inference run.

    Args:
        cfg: Pipeline config.
        run_id: Pipeline run id.
        split: ``"val"`` (one sweep setting) or ``"test"`` (the chosen setting).
        sweep_index: Index into the ``[evaluate]`` sweep settings, for val.
        dataset_id: The dataset whose split and ground truth pick the
            videos; ``run_id`` if None.

    Returns:
        The split's raw videos that have ground truth, the run directory,
        ``cfg.infer`` with the setting's gap fill and det_conf, and the
        dataset's key.
    """
    layout = Layout(cfg.path_to_base_output)
    dataset_id = layout.resolve_dataset(dataset_id, run_id)
    frozen = pl.read_parquet(layout.dataset_split_file(dataset_id))
    cameras = frozen.filter(pl.col("split") == split)["camera_id"].to_list()
    labeled = set(
        pl.read_parquet(
            layout.ground_truth(dataset_id), columns=["camera_id", "video_id"]
        ).iter_rows()
    )
    videos = [
        v
        for v in find_raw_videos(cfg.path_to_raw_videos, cameras)
        if parse_raw_video_path(v) in labeled
    ]
    if split == "val":
        assert sweep_index is not None
        gap, conf = cfg.sweep_settings()[sweep_index]
        run_dir = layout.sweep_infer_run(run_id, gap, conf).dir
    else:
        best = EvalRun(layout.evaluation_run("sweep", run_id)).best_setting
        setting = json.loads(best.read_text())
        gap, conf = setting["gap_fill_frames"], setting["det_conf"]
        run_dir = layout.test_infer_run(run_id).dir
    infer_cfg = dataclasses.replace(cfg.infer, gap_fill_frames=gap, det_conf=conf)
    return videos, run_dir, infer_cfg, dataset_key(layout, dataset_id)


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
    parser.add_argument(
        "--dataset", default=None, help="Dataset id (default: --run-id)"
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()
    if args.split and (args.run_id is None or args.cameras):
        parser.error("--split needs --run-id and no --camera")
    if (args.split == "val") != (args.sweep_index is not None):
        parser.error("--sweep-index goes with --split val")

    cfg = load_config(args.config_path, require_api_key=False)
    if args.split:
        videos, run_dir, infer_cfg, key = split_run(
            cfg, args.run_id, args.split, args.sweep_index, args.dataset
        )
    else:
        videos = find_raw_videos(cfg.path_to_raw_videos, args.cameras)
        layout = Layout(cfg.path_to_base_output)
        run_dir = layout.infer_run(default_run_id(args.run_id)).dir
        infer_cfg, key = cfg.infer, None
    videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No videos found under {cfg.path_to_raw_videos}")

    models = inference_models(cfg, train_run_id=args.train_run_id)
    if key and models.dataset and models.dataset["key"] != key:
        parser.error(
            f"Training run {args.train_run_id} used dataset "
            f"{models.dataset['id']}, which differs from this one"
        )
    print(f"Run directory: {run_dir}")
    failed = run_infer(
        videos=videos,
        run_dir=run_dir,
        models=models,
        cfg=infer_cfg,
        standard_fps=cfg.standard_fps,
        vocab=cfg.behaviors,
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
