"""Trainers for each model kind, looked up by ``ModelSpec.kind``.

Pipelines only ever talk to the :class:`Trainer` protocol, so swapping in a
different detector or classifier means writing one class here and pointing
``[train.detector]`` or ``[train.classifier]`` at its ``kind``. The
:func:`get_trainer` mapping is the one place model names live.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bird_cv.pipelines.config import ModelSpec


@dataclass(frozen=True)
class TrainContext:
    """Run-wide settings a trainer may need beyond its :class:`ModelSpec`.

    Attributes:
        device: Training device.
        select_conf: Detection confidence used when scoring detector
            checkpoints.
        iou_threshold: Minimum IoU for a detection hit.
    """

    device: int | str = 0
    select_conf: float = 0.25
    iou_threshold: float = 0.5


@dataclass(frozen=True)
class TrainResult:
    """Outcome of :meth:`Trainer.train` or :meth:`Trainer.tune`.

    Attributes:
        checkpoint: The selected checkpoint (file or directory).
        hyperparameters: Training arguments that produced it, reusable as
            ``train(..., hyperparameters=...)`` (e.g. for CV folds); empty
            means the trainer's defaults.
        metrics: Validation scores of the selected checkpoint.
    """

    checkpoint: Path
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


class Trainer(Protocol):
    """Trains one model kind from a :class:`ModelSpec`."""

    def checkpoint_path(self, run_dir: Path) -> Path:
        """Where :meth:`train` and :meth:`tune` put the selected checkpoint."""
        ...

    def train(
        self,
        spec: ModelSpec,
        dataset: Path,
        run_dir: Path,
        hyperparameters: dict[str, Any],
        context: TrainContext,
    ) -> TrainResult:
        """Train once with fixed hyperparameters, selecting on validation."""
        ...

    def tune(
        self, spec: ModelSpec, dataset: Path, run_dir: Path, context: TrainContext
    ) -> TrainResult:
        """Search ``spec.search_space`` and return the selected model."""
        ...

    def evaluate(
        self,
        checkpoint: Path,
        spec: ModelSpec,
        dataset: Path,
        split: str,
        output_dir: Path,
        context: TrainContext,
    ) -> dict[str, Any]:
        """Score a checkpoint on one split of its dataset (model-level metrics)."""
        ...


def build_search_space(spec: dict[str, dict]) -> dict[str, Any]:
    """Turn config distribution specs into Ray Tune search-space objects.

    Args:
        spec: Parameter name to ``{"type": ..., ...}``; types ``uniform``
            and ``loguniform`` (``lower``, ``upper``), ``randint``
            (``lower``, ``upper``, upper exclusive) and ``choice``
            (``values``).

    Returns:
        Parameter name to Ray Tune distribution.

    Raises:
        ValueError: On an unknown distribution type.
    """
    from ray import tune

    builders = {
        "uniform": lambda d: tune.uniform(d["lower"], d["upper"]),
        "loguniform": lambda d: tune.loguniform(d["lower"], d["upper"]),
        "randint": lambda d: tune.randint(d["lower"], d["upper"]),
        "choice": lambda d: tune.choice(d["values"]),
    }
    space = {}
    for name, dist in spec.items():
        if dist.get("type") not in builders:
            raise ValueError(
                f"Unknown distribution {dist.get('type')!r} for {name!r};"
                f" expected one of {sorted(builders)}"
            )
        space[name] = builders[dist["type"]](dist)
    return space


class YoloTrainer:
    """Ultralytics YOLO, selected on val presence F1 (ADR 0005)."""

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "best.pt"

    def _fit(
        self,
        spec: ModelSpec,
        dataset: Path,
        run_dir: Path,
        context: TrainContext,
        tune: bool,
        hyperparameters: dict[str, Any] | None,
    ) -> TrainResult:
        from bird_cv.detection.train_yolo import train_select_yolo

        best = train_select_yolo(
            dataset_root=dataset,
            run_dir=run_dir,
            model_path=spec.pretrained,
            epochs=spec.epochs,
            device=context.device,
            tune=tune,
            iterations=spec.tune_trials,
            hyperparameters=hyperparameters,
            select_conf=context.select_conf,
            iou_threshold=context.iou_threshold,
            search_space=build_search_space(spec.search_space) if tune else None,
            tune_options=spec.options if tune else None,
        )
        return TrainResult(
            checkpoint=Path(best.pop("best_checkpoint")),
            hyperparameters=best.pop("hyperparameters"),
            metrics=best,
        )

    def train(self, spec, dataset, run_dir, hyperparameters, context) -> TrainResult:
        return self._fit(spec, dataset, run_dir, context, False, hyperparameters)

    def tune(self, spec, dataset, run_dir, context) -> TrainResult:
        return self._fit(spec, dataset, run_dir, context, True, None)

    def evaluate(
        self, checkpoint, spec, dataset, split, output_dir, context
    ) -> dict[str, Any]:
        from bird_cv.detection.evaluate_yolo import yolo_presence_metrics

        return yolo_presence_metrics(
            dataset,
            checkpoint,
            split=split,
            conf=context.select_conf,
            iou_threshold=context.iou_threshold,
            device=context.device,
        )


class VideoMAETrainer:
    """VideoMAE clip classifier, selected on val window macro-F1 (ADR 0005).

    ``spec.options``: ``num_frames``, ``batch_size``, and for tuning
    ``cpus_per_trial`` and ``grace_period``.
    """

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "model" / "best"

    def train(self, spec, dataset, run_dir, hyperparameters, context) -> TrainResult:
        from bird_cv.classification.train_video_model import train_video_model

        opts = spec.options
        result = train_video_model(
            clips_root=dataset,
            output_root=run_dir,
            model_checkpoint=spec.pretrained,
            output_name="model",
            num_frames=opts.get("num_frames", 16),
            epochs=spec.epochs,
            batch_size=opts.get("batch_size", 8),
            device="cuda" if context.device != "cpu" else "cpu",
            selection_metric="macro_f1",
            **hyperparameters,
        )
        return TrainResult(
            checkpoint=self.checkpoint_path(run_dir),
            hyperparameters=dict(hyperparameters),
            metrics={
                "val_macro_f1": result["best_score"],
                "epoch": result["best_epoch"],
            },
        )

    def tune(self, spec, dataset, run_dir, context) -> TrainResult:
        from bird_cv.classification.tune_video_model import tune_video_model

        opts = spec.options
        best = tune_video_model(
            clips_root=dataset,
            output_root=run_dir / "tune",
            search_space=build_search_space(spec.search_space),
            model_checkpoint=spec.pretrained,
            num_samples=spec.tune_trials,
            epochs=spec.epochs,
            grace_period=opts.get("grace_period", 2),
            num_frames=opts.get("num_frames", 16),
            batch_size=opts.get("batch_size", 8),
            cpus_per_trial=opts.get("cpus_per_trial", 8),
        )
        # Retrain the best configuration once, saving its checkpoint.
        return self.train(spec, dataset, run_dir, best["hyperparameters"], context)

    def evaluate(
        self, checkpoint, spec, dataset, split, output_dir, context
    ) -> dict[str, Any]:
        from bird_cv.classification.evaluate_video_model import evaluate_video_model

        result = evaluate_video_model(
            clips_root=dataset,
            model_path=checkpoint,
            output_path=output_dir,
            num_frames=spec.options.get("num_frames", 16),
            split=split,
            window_macro_f1=True,
        )
        return {
            "macro_f1": result["macro_f1"],
            "n_skipped_out_of_vocabulary": result["n_skipped_out_of_vocabulary"],
        }


def get_trainer(kind: str) -> Trainer:
    """The trainer for a ``ModelSpec.kind``.

    Raises:
        ValueError: If no trainer is registered for ``kind``.
    """
    trainers: dict[str, type] = {"yolo": YoloTrainer, "videomae": VideoMAETrainer}
    if kind not in trainers:
        raise ValueError(f"No trainer for kind {kind!r}; known: {sorted(trainers)}")
    return trainers[kind]()
