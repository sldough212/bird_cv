import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from ray.tune import ResultGrid
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from bird_cv.core.layout import TrainRun
from bird_cv.detection.evaluate_yolo import yolo_presence_metrics
from bird_cv.training.ray_job import init_job_ray


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


def write_dataset_yaml(
    dataset_root: Path, path: Path, names: dict[int, str] | None = None
) -> Path:
    """Write an Ultralytics dataset YAML for a ``train``/``val`` dataset root.

    Args:
        dataset_root: Root holding ``images/{train,val}`` and
            ``labels/{train,val}`` (and ``test``, listed if present).
        path: Where to write the YAML.
        names: Class names; ``{0: "bird"}`` if None.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": names or {0: "bird"},
    }
    if (dataset_root / "images" / "test").is_dir():
        content["test"] = "images/test"
    with path.open("w") as f:
        yaml.dump(content, f, default_flow_style=False)
    return path


def tuned_hyperparameters(results: "ResultGrid", run: Path) -> dict[str, Any]:
    """The tuned values one Ultralytics run trained with, from its ``args.yaml``.

    Tuning trials' run directories are numbered by Ultralytics (``bird_yolo2``,
    ``bird_yolo3``, ...), not named after Ray trials, so the values are read
    from the run itself for the keys Ray sampled.

    Args:
        results: The tuning ``ResultGrid``; its configs name the tuned keys.
        run: The selected run directory.

    Returns:
        Tuned key to the value the run used.
    """
    keys = {k for result in results for k in (result.config or {}) if k != "data"}
    args = yaml.safe_load((run / "args.yaml").read_text())
    return {k: args[k] for k in sorted(keys)}


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
    names: dict[int, str] | None = None,
    score: Callable[[Path, Path], dict[str, Any]] | None = None,
    score_key: str = "f1",
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
    ``run_dir/best_hyperparameters.json`` so later runs can
    retrain with the same hyperparameters. Every run's scores go to
    ``run_dir/selection.parquet``, with ``within_run_selection`` and
    ``between_run_selection`` columns that record both criteria.

    Args:
        dataset_root: YOLO dataset root with ``train`` and ``val`` splits.
        run_dir: Run directory (already resolved).
        run_name: Training run name; tuning trials are ``{run_name}{n}``.
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
        names: Class names for the dataset YAML; ``{0: "bird"}`` if None.
        score: ``score(checkpoint, dataset_yaml)`` val metrics to select on,
            instead of presence metrics; higher ``score_key`` wins.
        score_key: The metric runs are selected on.

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

    model_config = write_dataset_yaml(
        dataset_root, run_dir / "model_config.yaml", names
    )
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
        if score is not None:
            metrics = score(checkpoint, model_config)
        else:
            metrics = yolo_presence_metrics(
                dataset_root,
                checkpoint,
                split="val",
                conf=select_conf,
                iou_threshold=iou_threshold,
                device=device,
            )
        candidate = checkpoint.parents[1].name
        print(f"{candidate}: val {score_key}={metrics[score_key]:.4f}")
        rows.append({"candidate": candidate, "checkpoint": str(checkpoint)} | metrics)
    if not rows:
        raise RuntimeError(f"No YOLO checkpoints found under {run_dir}")

    selection = (
        pl.DataFrame(rows)
        .with_columns(
            within_run_selection=pl.lit("ultralytics_fitness"),
            between_run_selection=pl.lit(
                f"val_{score_key}" if score else "val_presence_f1"
            ),
        )
        .sort(score_key, descending=True, nulls_last=True)
    )
    selection.write_parquet(TrainRun(run_dir).selection)
    best = selection.row(0, named=True)

    shutil.copy2(best["checkpoint"], run_dir / "best.pt")
    if tune:
        best_hyperparameters = tuned_hyperparameters(
            outcome,  # type: ignore[arg-type]
            Path(best["checkpoint"]).parents[1],
        )
    else:
        best_hyperparameters = dict(hyperparameters or {})
    TrainRun(run_dir).best_hyperparameters.write_text(
        json.dumps(best_hyperparameters, indent=2, default=float)
    )
    print(f"Selected {best['candidate']} (val {score_key}={best[score_key]:.4f})")
    return best | {
        "best_checkpoint": str(run_dir / "best.pt"),
        "hyperparameters": best_hyperparameters,
    }
