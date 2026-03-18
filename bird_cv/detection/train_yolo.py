from ultralytics import YOLO
from pathlib import Path


def train_yolo(
    output_root: Path,  # still creates a ton of files not in output_root / output_name
    output_name: str,
    iterations: int = 30,
    model_config: str = "detect_birds.yaml",
    model_path: str | Path = "yolo11n.pt",
    device: int | list[int] = -1,
    use_ray: bool = True,
    frozen_parameters: dict | None = None,
) -> None:
    """Tune YOLO model hyperparameters using Ultralytics' tuning interface.

    This function performs hyperparameter tuning for a YOLO model using
    Ultralytics' built-in mutation-based tuning, with optional integration
    of Ray Tune for advanced search strategies, parallel execution, and
    early stopping. Tuning optimizes the model's fitness score, which is a
    composite metric of precision, recall, and mAP.

    All outputs (logs, metrics, and model weights) are written under the
    specified project directory and run name.

    Args:
        output_root (Path): Root directory where tuning outputs will be saved.
            This corresponds to the Ultralytics `project` directory.
        output_name (str): Name of the tuning run. A subdirectory with this
            name will be created under `output_root`.
        iterations (int): Number of hyperparameter tuning iterations (trials)
            to run. Each iteration corresponds to a full training run.
        model_config (str): Path to the dataset configuration YAML file used
            for training (e.g., dataset splits, class names).
        model_path (str | Path): Path to a pretrained YOLO model checkpoint
            or model definition file to initialize training.
        device (int | list[int]): Device identifier(s) for training. Use `-1`
            for automatic device selection, an integer for a single GPU, or
            a list of integers for multi-GPU training.
        use_ray (bool): Whether to use Ray Tune for hyperparameter optimization.
            If False, Ultralytics' built-in tuner runs without Ray.
        frozen_parameters (dict | None): Dictionary of fixed training or tuning
            parameters to pass directly to `model.tune`. These parameters
            override defaults and are not mutated during tuning. If None,
            no additional parameters are frozen.
    """
    model = YOLO("yolo11n.pt")

    model.train(
        data=model_config,
        # use_ray=use_ray,
        # iterations=iterations,
        device=device,
        project=output_root,
        name=output_name,
        # **(frozen_parameters or {})
    )

    # Some tuning parameters left in here...


if __name__ == "__main__":
    train_yolo(
        output_root=Path("/gscratch/pdoughe1/yolo_train"),
        output_name="bird_test",
        model_config="bird_cv/detection/detect_birds.yaml",
        # model_path="/home/pdoughe1/Downloads/bird"
    )
