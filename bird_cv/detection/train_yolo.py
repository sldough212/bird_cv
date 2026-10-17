import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
from ultralytics import YOLO

import yaml
import tomli_w
import msgspec

from bird_cv.detection.evaluate_yolo import yolo_presence_metrics
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
    *,
    tune_kwargs: dict | None = None,
) -> Any:
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
        tune_kwargs: Extra arguments for Ultralytics' ``run_ray_tune`` only
            (e.g. ``gpu_per_trial``, ``grace_period``, ``space``). Ignored
            when ``tune=False``.

    Returns:
        The Ray Tune ``ResultGrid`` when tuning, else the training metrics
        object returned by ``model.train``.
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
        return model.tune(
            use_ray=True,
            iterations=iterations,
            **shared_kwargs,
            **(tune_kwargs or {}),
        )
    return model.train(**shared_kwargs)


# Ultralytics args that describe where/what a run trains, not how; dropped
# when reusing a selected run's hyperparameters on another dataset.
RUN_SPECIFIC_ARGS = frozenset(
    {"data", "project", "name", "device", "model", "save_dir", "resume"}
    | {"exist_ok", "mode", "task", "source", "tracker", "pretrained"}
)


def write_dataset_yaml(dataset_root: Path, path: Path) -> Path:
    """Write an Ultralytics dataset YAML for a ``train``/``val`` dataset root.

    Args:
        dataset_root: Root holding ``images/{train,val}`` and
            ``labels/{train,val}``.
        path: Where to write the YAML.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "bird"},
    }
    with path.open("w") as f:
        yaml.dump(content, f, default_flow_style=False)
    return path


def train_select_yolo(
    dataset_root: Path,
    run_dir: Path,
    run_name: str = "bird_yolo",
    model_path: str | Path = "yolo11n.pt",
    epochs: int = 60,
    device: int | str = 0,
    tune: bool = False,
    iterations: int = 30,
    hyperparameters: dict | None = None,
    select_conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Train (or Ray-tune) YOLO from generic weights and select on val presence F1.

    Every candidate checkpoint (one per tuning trial, or the single
    training run) is scored with
    :func:`bird_cv.detection.evaluate_yolo.yolo_presence_metrics` on the
    val split; the best is copied to ``run_dir/best.pt`` and its training
    arguments, minus :data:`RUN_SPECIFIC_ARGS`, to
    ``run_dir/best_hyperparameters.json`` so cross-validation folds can
    retrain with the same hyperparameters. All scores go to
    ``run_dir/selection.parquet``.

    Args:
        dataset_root: YOLO dataset root with ``train`` and ``val`` splits.
        run_dir: Run directory (already resolved).
        run_name: Training run name; tuning trials are ``{run_name}_{id}``.
        model_path: Initial weights. Defaults to generic ``yolo11n.pt``
            (ADR 0004).
        epochs: Epochs per training run or trial.
        device: Training device.
        tune: If ``True``, Ray Tune ``iterations`` trials on one GPU each.
        iterations: Number of tuning trials.
        hyperparameters: Fixed training arguments (e.g. a previous run's
            ``best_hyperparameters.json``); ``epochs`` here wins over
            ``epochs``.
        select_conf: Detection confidence used when scoring candidates.
        iou_threshold: Minimum IoU for a presence hit.

    Returns:
        ``best_checkpoint``, ``candidate`` (run directory name) and the
        best candidate's val presence metrics.
    """
    model_config = write_dataset_yaml(dataset_root, run_dir / "model_config.yaml")
    train_yolo(
        output_root=run_dir,
        output_name=run_name,
        model_config=str(model_config),
        model_path=model_path,
        device=device,
        tune=tune,
        iterations=iterations,
        frozen_parameters={"epochs": epochs} | (hyperparameters or {}),
        tune_kwargs={"gpu_per_trial": 1},
    )

    rows = []
    for checkpoint in sorted(run_dir.glob(f"{run_name}*/weights/best.pt")):
        metrics = yolo_presence_metrics(
            dataset_root,
            checkpoint,
            split="val",
            conf=select_conf,
            iou_threshold=iou_threshold,
            device=device,
        )
        candidate = checkpoint.parents[1].name
        print(f"{candidate}: val presence F1={metrics['f1']:.4f}")
        rows.append({"candidate": candidate, "checkpoint": str(checkpoint)} | metrics)
    if not rows:
        raise RuntimeError(f"No YOLO checkpoints found under {run_dir}")

    selection = pl.DataFrame(rows).sort("f1", descending=True, nulls_last=True)
    selection.write_parquet(run_dir / "selection.parquet")
    best = selection.row(0, named=True)

    shutil.copy2(best["checkpoint"], run_dir / "best.pt")
    with (Path(best["checkpoint"]).parents[1] / "args.yaml").open() as f:
        args = yaml.safe_load(f)
    best_hyperparameters = {k: v for k, v in args.items() if k not in RUN_SPECIFIC_ARGS}
    (run_dir / "best_hyperparameters.json").write_text(
        json.dumps(best_hyperparameters, indent=2)
    )
    print(f"Selected {best['candidate']} (val presence F1={best['f1']:.4f})")
    return best | {"best_checkpoint": str(run_dir / "best.pt")}


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
