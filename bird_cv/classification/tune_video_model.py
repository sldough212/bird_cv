"""Ray Tune + ASHA search over VideoMAE fine-tuning hyperparameters.

Each trial runs :func:`train_video_model` without saving checkpoints and
reports validation window macro-F1 (ADR 0005) after every epoch, so ASHA can
stop weak trials early. The best configuration is then retrained once with
checkpoint saving by the caller (see ``bird_cv/training/registry.py``).
"""

import json
from pathlib import Path
from typing import Any

from bird_cv.classification.train_video_model import train_video_model
from bird_cv.core.layout import TrainRun


def tune_video_model(
    clips_root: Path,
    output_root: Path,
    search_space: dict[str, Any],
    model_checkpoint: str | Path = "MCG-NJU/videomae-base",
    num_samples: int = 20,
    epochs: int = 10,
    grace_period: int = 2,
    num_frames: int = 16,
    batch_size: int = 8,
    cpus_per_trial: int = 8,
    gpus_per_trial: float = 1,
    train_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search VideoMAE hyperparameters on validation window macro-F1.

    Args:
        clips_root: Root with ``train/`` and ``val/`` clip splits (see
            :class:`bird_cv.classification.train_video_model.BehaviorClipDataset`).
        output_root: Directory for Ray Tune results; the best configuration
            is written to ``best_hyperparameters.json`` here.
        search_space: Ray Tune distributions keyed by
            :func:`train_video_model` keyword argument (e.g. ``lr``,
            ``freeze_encoder``, ``weight_decay``, ``class_weighted_loss``),
            from ``[models.classifier] search_space``.
        model_checkpoint: Generic pretrained weights to start every trial
            from (ADR 0004).
        num_samples: Number of trials. Defaults to 20.
        epochs: Maximum epochs per trial (ASHA's ``max_t``).
        grace_period: Epochs every trial runs before ASHA may stop it.
        num_frames: Frames per clip.
        batch_size: Dataloader batch size.
        cpus_per_trial: CPUs reserved per trial.
        gpus_per_trial: GPUs reserved per trial. Defaults to 1.
        train_kwargs: Further :func:`train_video_model` keyword arguments
            for every trial. They override ``num_frames``, ``epochs`` and
            ``batch_size``, and a sampled value overrides both, so any of
            them can be searched.

    Returns:
        ``hyperparameters`` (the best trial's value for each searched
        parameter), its ``val_macro_f1`` and best ``epoch``.
    """
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    clips_root = Path(clips_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_kwargs = {
        "num_frames": num_frames,
        "epochs": epochs,
        "batch_size": batch_size,
    } | (train_kwargs or {})
    max_epochs = base_kwargs["epochs"]

    def trainable(config: dict[str, Any]) -> None:
        train_video_model(
            clips_root=clips_root,
            output_root=output_root / "trials",
            model_checkpoint=model_checkpoint,  # type: ignore[arg-type]
            output_name=tune.get_context().get_trial_id(),
            selection_metric="macro_f1",
            save_checkpoint=False,
            epoch_callback=tune.report,
            **(base_kwargs | config),
        )

    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_macro_f1",
            mode="max",
            num_samples=num_samples,
            scheduler=ASHAScheduler(
                time_attr="epoch",
                max_t=max_epochs,
                grace_period=min(grace_period, max_epochs),
                reduction_factor=3,
            ),
        ),
        run_config=tune.RunConfig(storage_path=str(output_root), name="ray"),
    )
    results = tuner.fit()
    results.get_dataframe().to_csv(TrainRun(output_root).trials, index=False)

    # Best epoch of any trial, not just each trial's last reported epoch.
    best = results.get_best_result(metric="val_macro_f1", mode="max", scope="all")
    history = best.metrics_dataframe
    best_epoch = history.loc[history["val_macro_f1"].idxmax()]
    summary = {
        "hyperparameters": {k: best.config[k] for k in search_space},
        "val_macro_f1": float(best_epoch["val_macro_f1"]),
        "epoch": int(best_epoch["epoch"]),
    }
    TrainRun(output_root).best_hyperparameters.write_text(json.dumps(summary, indent=2))
    print(f"Best VideoMAE config: {summary}")
    return summary
