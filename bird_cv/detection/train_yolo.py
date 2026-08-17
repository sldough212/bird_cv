from pathlib import Path

from ultralytics import YOLO

import yaml
import tomli_w
import msgspec

from bird_cv.pipelines.config import resolve_run_dir


class Paths(msgspec.Struct):
    base_path: str
    model_config: str
    pretrained_checkpoint: str
    output_root: str = ""
    new_checkpoint: str = ""


class Training(msgspec.Struct):
    epochs: int = 30
    device: int = 0
    tune: bool = False
    tune_iterations: int = 30
    run_name: str = "bird_yolo"


class DetectionConfig(msgspec.Struct):
    paths: Paths
    training: Training


def train_yolo(
    output_root: Path,
    output_name: str,
    model_config: str = "detect_birds.yaml",
    model_path: str | Path = "yolo11n.pt",
    device: int | str = -1,
    tune: bool = False,
    iterations: int = 30,
    frozen_parameters: dict | None = None,
) -> None:
    """Train or hyperparameter-tune a YOLO model.

    Args:
        output_root: Root directory where outputs will be saved.
        output_name: Name of the run. A subdirectory with this name will be
            created under ``output_root``.
        model_config: Path to the dataset configuration YAML file (dataset
            splits, class names).
        model_path: Path to a pretrained YOLO checkpoint or model definition
            file to initialise from.
        device: Device identifier(s) for training. Use ``-1`` for automatic
            selection, an integer for a single GPU, or a list for multi-GPU.
        tune: If ``True``, run a Ray Tune hyperparameter search instead of a
            standard training run.
        iterations: Number of tuning trials. Only used when ``tune=True``.
        frozen_parameters: Fixed training parameters passed directly to
            ``model.train`` or ``model.tune``. These are not mutated during
            tuning. If ``None``, no additional parameters are fixed.
    """
    model = YOLO(model_path)

    shared_kwargs = dict(
        data=model_config,
        device=device,
        project=str(output_root),
        name=output_name,
        **(frozen_parameters or {}),
    )

    if tune:
        model.tune(
            use_ray=True,
            iterations=iterations,
            **shared_kwargs,
        )
    else:
        model.train(**shared_kwargs)


def update_train_yolo(
    base_path: Path,
    previous_model_config: Path,
    pretrained_checkpoint: Path,
    yolo_training_data: Path,
    epochs: int = 30,
    device: int | str = 0,
    tune: bool = False,
    tune_iterations: int = 30,
    run_name: str = "bird_yolo",
) -> None:
    """Train YOLO into a new run directory and save the resolved config.

    Args:
        base_path: Root directory under which a new run directory is created.
        previous_model_config: Dataset config YAML to copy and repoint at
            ``yolo_training_data``.
        pretrained_checkpoint: YOLO checkpoint or model definition to
            initialize from.
        yolo_training_data: Root directory of the YOLO training data.
        epochs: Number of training epochs. Defaults to 30.
        device: Device identifier(s) for training. Defaults to 0.
        tune: If ``True``, run a Ray Tune hyperparameter search. Defaults
            to ``False``.
        tune_iterations: Number of tuning trials. Only used when
            ``tune=True``. Defaults to 30.
        run_name: Name of the run subdirectory. Defaults to ``"bird_yolo"``.
    """

    # Build the yolo config
    paths = Paths(
        base_path=str(base_path),
        model_config=str(previous_model_config),
        pretrained_checkpoint=str(pretrained_checkpoint),
    )
    training = Training(
        epochs=epochs,
        device=device,
        tune=tune,
        tune_iterations=tune_iterations,
        run_name=run_name,
    )
    cfg = DetectionConfig(
        paths=paths,
        training=training,
    )
    run_dir = resolve_run_dir(Path(base_path), None)

    # Copy and update the config yaml
    previous_yaml_file = previous_model_config
    with previous_yaml_file.open("r") as f:
        previous_yaml = yaml.safe_load(f)

    previous_yaml["path"] = str(yolo_training_data)
    with (run_dir / "model_config.yaml").open("w") as f:
        yaml.dump(previous_yaml, f, default_flow_style=False)

    cfg.paths.model_config = str(run_dir / "model_config.yaml")
    cfg.paths.output_root = str(run_dir)
    cfg.paths.new_checkpoint = str(
        run_dir / cfg.training.run_name / "weights" / "best.pt"
    )

    # Save the config in run_dir
    with open(run_dir / "run_config.toml", "wb") as f:
        tomli_w.dump(msgspec.to_builtins(cfg), f)

    train_yolo(
        output_root=run_dir,
        output_name=cfg.training.run_name,
        model_config=cfg.paths.model_config,
        model_path=cfg.paths.pretrained_checkpoint,
        device=cfg.training.device,
        tune=cfg.training.tune,
        iterations=cfg.training.tune_iterations,
        frozen_parameters={"epochs": cfg.training.epochs},
    )


if __name__ == "__main__":
    train_yolo(
        output_root=Path("/gscratch/pdoughe1/20260331_194037/training"),
        output_name="bird_test",
        model_config="bird_cv/detection/detect_birds.yaml",
        model_path="yolo11n.pt",
        frozen_parameters={"epochs": 30},
    )
