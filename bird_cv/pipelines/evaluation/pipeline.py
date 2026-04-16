"""Evaluation pipeline: YOLO tracking + MOT metrics + VideoMAE classification."""

from pathlib import Path

import msgspec

from bird_cv.classification.evaluate_video_model import evaluate_video_model
from bird_cv.detection.evaluate_yolo import run_tracking_on_test
from bird_cv.detection.mot_evaluate import evaluate_tracking
from bird_cv.pipelines.config import load_config, resolve_run_dir


class Run(msgspec.Struct):
    base_path: str
    run_id: str = ""


class Paths(msgspec.Struct):
    yolo_crop_path: str
    video_crop_path: str
    yolo_model_path: str
    videomae_model_path: str


class Detection(msgspec.Struct):
    num_frames: int = 16


class Classification(msgspec.Struct):
    num_frames: int = 16
    batch_size: int = 8
    device: str = "cuda"


class EvaluationConfig(msgspec.Struct):
    run: Run
    paths: Paths
    detection: Detection
    classification: Classification


def run_evaluation_pipeline(config_path: Path) -> None:
    """Run detection tracking, MOT evaluation, and VideoMAE classification evaluation.

    Args:
        config_path: Path to the pipeline TOML config file.
    """
    cfg = load_config(config_path, EvaluationConfig)

    run_dir = resolve_run_dir(Path(cfg.run.base_path), cfg.run.run_id or None)
    eval_dir = run_dir / "evaluation"

    yolo_crop_path = Path(cfg.paths.yolo_crop_path)
    video_crop_path = Path(cfg.paths.video_crop_path)
    tracking_output = eval_dir / "tracking"

    # Step 1: Run YOLO tracking on test set
    print("\n=== Step 1: Running YOLO tracking ===")
    run_tracking_on_test(
        yolo_crop_path=yolo_crop_path,
        output_root=tracking_output,
        model_path=cfg.paths.yolo_model_path,
    )

    # Step 2: MOT evaluation
    print("\n=== Step 2: MOT evaluation ===")
    evaluate_tracking(
        yolo_crop_path=yolo_crop_path,
        tracking_output_root=tracking_output,
        output_path=eval_dir / "mot_results.parquet",
    )

    # Step 3: VideoMAE classification evaluation
    print("\n=== Step 3: VideoMAE evaluation ===")
    evaluate_video_model(
        clips_root=video_crop_path,
        model_path=Path(cfg.paths.videomae_model_path),
        output_path=eval_dir / "videomae",
        num_frames=cfg.classification.num_frames,
        batch_size=cfg.classification.batch_size,
        device=cfg.classification.device,
    )

    print(f"\nEvaluation complete. Outputs at: {eval_dir}")


if __name__ == "__main__":
    run_evaluation_pipeline(config_path=Path(__file__).parent / "config.toml")
