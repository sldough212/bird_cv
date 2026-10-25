import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
from ultralytics import YOLO

if TYPE_CHECKING:
    from ray.tune import ResultGrid
    from ultralytics.utils.metrics import DetMetrics

import yaml
import tomli_w
import msgspec

from bird_cv.core.layout import TrainRun
from bird_cv.training.ray_job import init_job_ray
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
    frozen_parameters: dict[str, Any] | None = None,
    *,
    tune_kwargs: dict[str, Any] | None = None,
) -> "ResultGrid | DetMetrics | None":
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
            (e.g. ``gpu_per_trial``, ``grace_period``, ``space``).
            ``gpu_per_trial`` defaults to 1. Ignored when ``tune=False``.

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
        init_job_ray()
        return model.tune(
            use_ray=True,
            iterations=iterations,
            **shared_kwargs,
            # Ray gives a trial no GPU unless asked (gpu_per_trial or 0).
            **({"gpu_per_trial": 1} | (tune_kwargs or {})),
        )
    return model.train(**shared_kwargs)


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


def _trial_hyperparameters(results: "ResultGrid", run_name: str) -> dict[str, dict]:
    """Sampled hyperparameters per tuning run directory name.

    Ultralytics names each trial's run ``{run_name}_{last part of trial id}``
    and adds ``data`` to every config; that key is dropped.
    """
    by_run = {}
    for result in results:
        suffix = Path(result.path).name.split("_")[-1]
        config = {k: v for k, v in (result.config or {}).items() if k != "data"}
        by_run[f"{run_name}_{suffix}"] = config
    return by_run


def train_select_yolo(
    dataset_root: Path,
    run_dir: Path,
    run_name: str = "bird_yolo",
    model_path: str | Path = "yolo11n.pt",
    epochs: int = 60,
    device: int | str = 0,
    tune: bool = False,
    iterations: int = 30,
    hyperparameters: dict[str, Any] | None = None,
    select_conf: float = 0.25,
    iou_threshold: float = 0.5,
    search_space: dict[str, Any] | None = None,
    train_options: dict[str, Any] | None = None,
    tune_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train (or Ray-tune) YOLO from generic weights and select on val presence F1.

    Selection has two levels. Within each run (one per tuning trial, or the
    single training run), Ultralytics keeps the epoch with the best
    Ultralytics fitness (mAP-based) as that run's ``weights/best.pt``;
    presence F1 is not computed per epoch. Between runs, each run's
    ``best.pt`` is scored with
    :func:`bird_cv.detection.evaluate_yolo.yolo_presence_metrics` on the
    val split, and the highest presence F1 wins. The winner is copied to
    ``run_dir/best.pt`` and its hyperparameters (the tuned values sampled
    for that trial, or the fixed ``hyperparameters`` given) go to
    ``run_dir/best_hyperparameters.json`` so cross-validation folds can
    retrain with the same hyperparameters. Every run's scores go to
    ``run_dir/selection.parquet``, with ``within_run_selection`` and
    ``between_run_selection`` columns that record both criteria.

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
        search_space: Ray Tune search space for tuning; ``None`` or empty
            uses Ultralytics' default space.
        train_options: Ultralytics training arguments for every run, tuned
            or not (e.g. ``imgsz``, ``batch``); ``hyperparameters`` override
            them.
        tune_options: Extra Ultralytics ``run_ray_tune`` arguments, used
            only when tuning (e.g. ``grace_period``).

    Returns:
        ``best_checkpoint``, ``candidate`` (run directory name),
        ``hyperparameters`` and the best candidate's val presence metrics.

    Raises:
        ValueError: If ``tune`` and ``hyperparameters`` are both given, or
            ``run_dir`` already holds ``{run_name}*`` runs (only runs made
            by this call are scored).
    """
    if tune and hyperparameters:
        raise ValueError("Pass either tune=True or fixed hyperparameters, not both")
    existing = sorted(p.name for p in run_dir.glob(f"{run_name}*"))
    if existing:
        raise ValueError(
            f"{run_dir} already holds YOLO runs {existing}; use a fresh run dir"
        )

    model_config = write_dataset_yaml(dataset_root, run_dir / "model_config.yaml")
    outcome = train_yolo(
        output_root=run_dir,
        output_name=run_name,
        model_config=str(model_config),
        model_path=model_path,
        device=device,
        tune=tune,
        iterations=iterations,
        frozen_parameters={"epochs": epochs}
        | (train_options or {})
        | (hyperparameters or {}),
        tune_kwargs={"gpu_per_trial": 1, "space": search_space or {}}
        | (tune_options or {}),
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

    selection = (
        pl.DataFrame(rows)
        .with_columns(
            within_run_selection=pl.lit("ultralytics_fitness"),
            between_run_selection=pl.lit("val_presence_f1"),
        )
        .sort("f1", descending=True, nulls_last=True)
    )
    selection.write_parquet(TrainRun(run_dir).selection)
    best = selection.row(0, named=True)

    shutil.copy2(best["checkpoint"], run_dir / "best.pt")
    if tune:
        trials = _trial_hyperparameters(outcome, run_name)  # type: ignore[arg-type]
        if best["candidate"] not in trials:
            raise RuntimeError(
                f"No tuning config found for {best['candidate']}; known: {sorted(trials)}"
            )
        best_hyperparameters = trials[best["candidate"]]
    else:
        best_hyperparameters = dict(hyperparameters or {})
    TrainRun(run_dir).best_hyperparameters.write_text(
        json.dumps(best_hyperparameters, indent=2, default=float)
    )
    print(f"Selected {best['candidate']} (val presence F1={best['f1']:.4f})")
    return best | {
        "best_checkpoint": str(run_dir / "best.pt"),
        "hyperparameters": best_hyperparameters,
    }


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
