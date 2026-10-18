"""Model kinds, looked up by ``ModelSpec.kind``: how to train and load each.

Pipelines only ever talk to the :class:`Trainer` protocol, so swapping in a
different detector or classifier means writing one class here and pointing
``[train.detector]`` or ``[train.classifier]`` at its ``kind``; training
and inference (:meth:`Trainer.load_for_inference`) then both use it.
:func:`_trainers` is the one place detector and classifier names live (the
segmenter, which is never trained, is in :mod:`bird_cv.inference.registry`).
Importing this module doesn't import Ray or any model library; config
loading uses it to validate kinds and search spaces.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
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

    def dataset_cameras(self, dataset: Path) -> set[str]:
        """Cameras visible in a dataset's train and val data, from its layout.

        Used to catch CV fold leaks; empty if the layout doesn't show them.
        """
        ...

    def train(
        self,
        spec: "ModelSpec",
        dataset: Path,
        run_dir: Path,
        hyperparameters: dict[str, Any],
        context: TrainContext,
    ) -> TrainResult:
        """Train once with fixed hyperparameters, selecting on validation."""
        ...

    def tune(
        self, spec: "ModelSpec", dataset: Path, run_dir: Path, context: TrainContext
    ) -> TrainResult:
        """Search ``spec.search_space`` and return the selected model."""
        ...

    def evaluate(
        self,
        checkpoint: Path,
        spec: "ModelSpec",
        dataset: Path,
        split: str,
        output_dir: Path,
        context: TrainContext,
    ) -> dict[str, Any]:
        """Score a checkpoint on one split of its dataset (model-level metrics)."""
        ...

    def infer_option_keys(self) -> set[str]:
        """``infer_options`` this kind accepts (checked at config load)."""
        ...

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """The inference model for a checkpoint, built with ``spec.infer_options``.

        A detector kind returns a
        :class:`~bird_cv.inference.protocols.StreamingDetector`, a
        classifier kind a :class:`~bird_cv.inference.protocols.FrameClassifier`.
        """
        ...


def _distribution_keys() -> dict[str, tuple[str, ...]]:
    """Required keys for each search-space distribution type."""
    return {
        "uniform": ("lower", "upper"),
        "loguniform": ("lower", "upper"),
        "randint": ("lower", "upper"),
        "choice": ("values",),
    }


def check_search_space(spec: dict[str, dict], where: str = "search_space") -> None:
    """Validate config distribution specs without importing Ray.

    Args:
        spec: Parameter name to ``{"type": ..., ...}`` (see
            :func:`build_search_space`).
        where: Location named in error messages, e.g. ``"[train.classifier]"``.

    Raises:
        ValueError: On a non-table entry, an unknown ``type`` or a missing
            required key.
    """
    required = _distribution_keys()
    for name, dist in spec.items():
        if not isinstance(dist, dict):
            raise ValueError(f"{where} search_space.{name} must be a table")
        kind = dist.get("type")
        if kind not in required:
            raise ValueError(
                f"{where} search_space.{name} has type {kind!r};"
                f" expected one of {sorted(required)}"
            )
        missing = [k for k in required[kind] if k not in dist]
        if missing:
            raise ValueError(
                f"{where} search_space.{name} ({kind}) is missing {missing}"
            )


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
        ValueError: As for :func:`check_search_space`.
    """
    check_search_space(spec)
    from ray import tune

    builders = {
        "uniform": lambda d: tune.uniform(d["lower"], d["upper"]),
        "loguniform": lambda d: tune.loguniform(d["lower"], d["upper"]),
        "randint": lambda d: tune.randint(d["lower"], d["upper"]),
        "choice": lambda d: tune.choice(d["values"]),
    }
    return {name: builders[dist["type"]](dist) for name, dist in spec.items()}


class YoloTrainer:
    """Ultralytics YOLO, selected on val presence F1 (ADR 0005).

    ``spec.options`` are Ultralytics training arguments applied to every
    run; ``spec.tune_options`` go to Ultralytics' ``run_ray_tune`` only
    (e.g. ``grace_period``).
    """

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "best.pt"

    def dataset_cameras(self, dataset: Path) -> set[str]:
        """Camera folders under ``images/{train,val}/``."""
        return {
            d.name
            for split in ("train", "val")
            if (dataset / "images" / split).is_dir()
            for d in (dataset / "images" / split).iterdir()
            if d.is_dir()
        }

    def _fit(
        self,
        spec: "ModelSpec",
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
            train_options=spec.options,
            tune_options=spec.tune_options if tune else None,
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

    def infer_option_keys(self) -> set[str]:
        """:class:`YoloDetector` fields; ``tracker_path`` is required."""
        return {"tracker_path", "conf", "iou", "max_det"}

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """A :class:`YoloDetector`; ``infer_options`` are its fields."""
        from bird_cv.detection.track import YoloDetector

        if "tracker_path" not in spec.infer_options:
            raise ValueError("yolo needs infer_options.tracker_path")
        return YoloDetector(model_path=checkpoint, **spec.infer_options)


class VideoMAETrainer:
    """VideoMAE clip classifier, selected on val window macro-F1 (ADR 0005).

    ``spec.options`` are :func:`train_video_model` keyword arguments for
    every run (e.g. ``num_frames``, ``batch_size``); ``spec.tune_options``
    are :func:`tune_video_model` arguments (``cpus_per_trial``,
    ``grace_period``).
    """

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "model" / "best"

    def dataset_cameras(self, dataset: Path) -> set[str]:
        """Clip paths don't carry the camera; rely on ``split.parquet``."""
        return set()

    def _train_kwargs(
        self, spec: "ModelSpec", context: TrainContext, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """One kwargs dict: run settings, then options, then hyperparameters."""
        return (
            {
                "epochs": spec.epochs,
                "device": "cpu" if context.device == "cpu" else "cuda",
            }
            | spec.options
            | overrides
        )

    def train(self, spec, dataset, run_dir, hyperparameters, context) -> TrainResult:
        from bird_cv.classification.train_video_model import train_video_model

        result = train_video_model(
            clips_root=dataset,
            output_root=run_dir,
            model_checkpoint=spec.pretrained,
            output_name="model",
            selection_metric="macro_f1",
            **self._train_kwargs(spec, context, hyperparameters),
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

        best = tune_video_model(
            clips_root=dataset,
            output_root=run_dir / "tune",
            search_space=build_search_space(spec.search_space),
            model_checkpoint=spec.pretrained,
            num_samples=spec.tune_trials,
            train_kwargs=self._train_kwargs(spec, context, {}),
            **spec.tune_options,
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

    def infer_option_keys(self) -> set[str]:
        """:class:`VideoMAEClassifier` fields other than the checkpoint."""
        return {"num_frames", "stride", "batch_size"}

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """A :class:`VideoMAEClassifier`; ``infer_options`` are its fields.

        ``num_frames`` follows training (``spec.options``) unless set there.
        """
        from bird_cv.classification.predict_video_model import VideoMAEClassifier

        trained = {k: v for k, v in spec.options.items() if k == "num_frames"}
        return VideoMAEClassifier(
            model_path=str(checkpoint), **(trained | spec.infer_options)
        )


def _trainers() -> dict[str, type]:
    """The registry: ``ModelSpec.kind`` to trainer class."""
    return {"yolo": YoloTrainer, "videomae": VideoMAETrainer}


def trainer_kinds() -> list[str]:
    """Every registered ``kind``, sorted."""
    return sorted(_trainers())


def get_trainer(kind: str) -> Trainer:
    """The trainer for a ``ModelSpec.kind``.

    Raises:
        ValueError: If no trainer is registered for ``kind``.
    """
    trainers = _trainers()
    if kind not in trainers:
        raise ValueError(f"No trainer for kind {kind!r}; known: {trainer_kinds()}")
    return trainers[kind]()
