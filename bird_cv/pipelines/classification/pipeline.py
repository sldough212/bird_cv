"""Classification pipeline: fine-tune VideoMAE on behavior clips."""

from pathlib import Path

import msgspec

from bird_cv.classification.train_video_model import train_video_model
from bird_cv.pipelines.config import load_config, resolve_run_dir


class Run(msgspec.Struct):
    base_path: str
    run_id: str = ""


class Paths(msgspec.Struct):
    video_crop_path: str
    model_checkpoint: str


class Training(msgspec.Struct):
    num_frames: int = 16
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    device: str = "cuda"
    freeze_encoder: bool = True
    run_name: str = "videomae_behavior"


class ClassificationConfig(msgspec.Struct):
    run: Run
    paths: Paths
    training: Training


def run_classification_pipeline(config_path: Path) -> None:
    """Fine-tune a VideoMAE model for behavior classification.

    Args:
        config_path: Path to the pipeline TOML config file.
    """
    cfg = load_config(config_path, ClassificationConfig)

    run_dir = resolve_run_dir(Path(cfg.run.base_path), cfg.run.run_id or None)

    print("\n=== Training VideoMAE ===")
    train_video_model(
        clips_root=Path(cfg.paths.video_crop_path),
        output_root=run_dir / "training",
        output_name=cfg.training.run_name,
        model_checkpoint=cfg.paths.model_checkpoint,
        num_frames=cfg.training.num_frames,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        device=cfg.training.device,
        freeze_encoder=cfg.training.freeze_encoder,
    )

    print(f"\nClassification training complete. Outputs at: {run_dir / 'training'}")


if __name__ == "__main__":
    run_classification_pipeline(config_path=Path(__file__).parent / "config.toml")
