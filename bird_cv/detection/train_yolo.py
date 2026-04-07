from pathlib import Path

from ultralytics import YOLO


def train_yolo(
    output_root: Path,
    output_name: str,
    model_config: str = "detect_birds.yaml",
    model_path: str | Path = "yolo11n.pt",
    device: int | list[int] = -1,
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


if __name__ == "__main__":
    train_yolo(
        output_root=Path("/gscratch/pdoughe1/yolo_train"),
        output_name="bird_test",
        model_config="bird_cv/detection/detect_birds.yaml",
        model_path="yolo11n.pt",
        frozen_parameters={"epochs": 30},
    )
