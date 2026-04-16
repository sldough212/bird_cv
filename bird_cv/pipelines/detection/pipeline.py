"""Detection pipeline: train YOLO on cropped cage images."""

from pathlib import Path

import msgspec

from bird_cv.detection.train_yolo import train_yolo
from bird_cv.pipelines.config import load_config, resolve_run_dir


class Run(msgspec.Struct):
    base_path: str
    run_id: str = ""


class Paths(msgspec.Struct):
    yolo_crop_path: str
    model_config: str
    pretrained_checkpoint: str


class Training(msgspec.Struct):
    epochs: int = 30
    device: int = 0
    tune: bool = False
    tune_iterations: int = 30
    run_name: str = "bird_yolo"


class DetectionConfig(msgspec.Struct):
    run: Run
    paths: Paths
    training: Training


def run_detection_pipeline(config_path: Path) -> None:
    """Train a YOLO detection model on cropped cage images.

    Args:
        config_path: Path to the pipeline TOML config file.
    """
    cfg = load_config(config_path, DetectionConfig)

    run_dir = resolve_run_dir(Path(cfg.run.base_path), cfg.run.run_id or None)

    print("\n=== Training YOLO ===")
    train_yolo(
        output_root=run_dir / "training",
        output_name=cfg.training.run_name,
        model_config=cfg.paths.model_config,
        model_path=cfg.paths.pretrained_checkpoint,
        device=cfg.training.device,
        tune=cfg.training.tune,
        iterations=cfg.training.tune_iterations,
        frozen_parameters={"epochs": cfg.training.epochs},
    )

    print(f"\nDetection training complete. Outputs at: {run_dir / 'training'}")


if __name__ == "__main__":
    run_detection_pipeline(config_path=Path(__file__).parent / "config.toml")
