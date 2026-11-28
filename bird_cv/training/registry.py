"""Model kinds, looked up by ``ModelSpec.kind``: how to train and load each.

Pipelines only ever talk to the :class:`Trainer` protocol, so swapping in a
different detector or classifier means writing one class here and pointing
``[models.detector]`` or ``[models.classifier]`` at its ``kind``; training
and inference (:meth:`Trainer.load_for_inference`) then both use it.
:func:`_trainers` is the one place detector and classifier names live (the
segmenter, which is never trained, is in :mod:`bird_cv.inference.registry`).
Config loading uses it to validate kinds and search spaces.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from bird_cv.core.layout import TrainRun
from ray import tune
from bird_cv.detection.train_yolo import train_select_yolo
from bird_cv.detection.evaluate_yolo import yolo_presence_metrics
from bird_cv.detection.track import YoloDetector
from bird_cv.detection.rf_detr import (
    RfdetrDetector,
    rfdetr_presence_metrics,
    train_rfdetr,
)
import logging

import polars as pl
from bird_cv.core.frames import canonical_to_ls
from bird_cv.preprocessing.crop import ground_truth_crop_centers
from bird_cv.preprocessing.annotations_to_yolo import stream_annotations_to_yolo
from bird_cv.classification.train_video_model import train_video_model
from bird_cv.classification.tune_video_model import tune_video_model
from bird_cv.classification.evaluate_video_model import evaluate_video_model
from bird_cv.classification.predict_video_model import VideoClassifier
from bird_cv.preprocessing.annotations_to_mae import (
    label_dir_name,
    stream_annotations_to_mae,
)

from bird_cv.core.layout import Layout

if TYPE_CHECKING:
    from bird_cv.pipelines.config import ModelSpec

logger = logging.getLogger(__name__)


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
            ``train(..., hyperparameters=...)``; empty means the trainer's
            defaults.
        metrics: Validation scores of the selected checkpoint.
    """

    checkpoint: Path
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMedia:
    """Exports a dataset is built from; each task's ``data.video`` is its media.

    Attributes:
        detect_export: The snapshot's interpolated Detect export (cage videos).
        classify_export: The snapshot's Classify export (bird-crop videos).
        standard_fps: Frame rate of the standardized videos.
        processes: Worker processes for the converters.
    """

    detect_export: Path
    classify_export: Path
    standard_fps: float
    processes: int = 1


class Trainer(Protocol):
    """Trains one model kind from a :class:`ModelSpec`."""

    # Training data format; kinds sharing one share a dataset directory
    data_format: str

    def checkpoint_path(self, run_dir: Path) -> Path:
        """Where :meth:`train` and :meth:`tune` put the selected checkpoint."""
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

    def check_spec(self, spec: "ModelSpec", where: str) -> None:
        """Raise ``ValueError`` on options this kind can't use (at config load)."""
        ...

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """The inference model for a checkpoint, built with ``spec.infer_options``.

        A detector kind returns a
        :class:`~bird_cv.inference.protocols.StreamingDetector`, a
        classifier kind a :class:`~bird_cv.inference.protocols.FrameClassifier`.
        """
        ...

    def prepared_selection(self, layout: Layout, dataset_id: str) -> Path:
        """The frame or clip list prepare sampled for this kind (frozen split)."""
        ...

    def dataset_export(self, media: DatasetMedia, spec: "ModelSpec") -> Path:
        """The export whose tasks' media :meth:`build_dataset` reads."""
        ...

    def build_dataset(
        self,
        selection: "pl.DataFrame",
        ground_truth: "pl.DataFrame",
        spec: "ModelSpec",
        media: DatasetMedia,
        out_dir: Path,
        guidance_path: Path,
    ) -> None:
        """Write the training data :meth:`train` reads for ``selection``.

        Writes the converter's split-guidance parquet to ``guidance_path``.
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
        where: Location named in error messages, e.g. ``"[models.classifier]"``.

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

    data_format = "yolo"

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "best.pt"

    def _fit(
        self,
        spec: "ModelSpec",
        dataset: Path,
        run_dir: Path,
        context: TrainContext,
        tune: bool,
        hyperparameters: dict[str, Any] | None,
    ) -> TrainResult:
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

    def check_spec(self, spec, where) -> None:
        return None

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """A :class:`YoloDetector`; ``infer_options`` are its fields."""
        if "tracker_path" not in spec.infer_options:
            raise ValueError("yolo needs infer_options.tracker_path")
        return YoloDetector(model_path=checkpoint, **spec.infer_options)

    def prepared_selection(self, layout, dataset_id) -> Path:
        return layout.yolo_frames(dataset_id)

    def dataset_export(self, media, spec):
        return media.detect_export

    def build_dataset(
        self, selection, ground_truth, spec, media, out_dir, guidance_path
    ) -> None:
        """Full cage-video frames and labels, ``images|labels/{split}/``.

        One guidance row per cage video (``by_cage``), its ``target_frames``
        the selected canonical frames as 1-indexed video frames.
        """
        guidance = (
            selection.group_by("camera_id", "video_id", "cage_id", "split")
            .agg(target_frames=(pl.col("frame") + canonical_to_ls(0)).sort())
            .with_columns(
                video_path=pl.col("camera_id") + "/" + pl.col("video_id") + ".mp4",
                # Ground truth keeps only standardized-fps cages:
                # LS frame = canonical + 1
                fps=pl.lit(float(media.standard_fps)),
            )
        )
        guidance.write_parquet(guidance_path)
        stream_annotations_to_yolo(
            path_to_videos=Path(),  # unused: media come from the export
            path_to_annotations=media.detect_export,
            path_to_output=out_dir,
            path_to_guidance=guidance_path,
            processes=media.processes,
            by_cage=True,
        )
        expected = selection.select(
            "split",
            stem=pl.format(
                "{}_{}_{}_{}",
                "camera_id",
                "video_id",
                "cage_id",
                (pl.col("frame") + canonical_to_ls(0)).cast(pl.String).str.zfill(5),
            ),
        )
        missing = [
            f"{split}/{stem}"
            for split, stem in expected.iter_rows()
            if not (out_dir / "images" / split / f"{stem}.jpg").exists()
            or not (out_dir / "labels" / split / f"{stem}.txt").exists()
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {expected.height} selected frames were not "
                f"written to {out_dir}, e.g. {missing[:10]}"
            )


class VideoMAETrainer:
    """VideoMAE clip classifier, selected on val window macro-F1 (ADR 0005).

    ``spec.options`` are :func:`train_video_model` keyword arguments for
    every run (e.g. ``num_frames``, ``batch_size``, ``input_size``);
    ``spec.tune_options`` are :func:`tune_video_model` arguments
    (``cpus_per_trial``, ``grace_period``). ``options.input`` picks
    bird-crop or cage clips (ADR 0009) and isn't passed on.
    """

    data_format = "videomae"
    backbone = "videomae"

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "model" / "best"

    def _train_kwargs(
        self, spec: "ModelSpec", context: TrainContext, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """One kwargs dict: run settings, then options, then hyperparameters."""
        return (
            {
                "epochs": spec.epochs,
                "device": "cpu" if context.device == "cpu" else "cuda",
                "backbone": self.backbone,
                "input_size": spec.input_size,
            }
            | {k: v for k, v in spec.options.items() if k != "input"}
            | overrides
        )

    def train(self, spec, dataset, run_dir, hyperparameters, context) -> TrainResult:
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
        best = tune_video_model(
            clips_root=dataset,
            output_root=TrainRun(run_dir).tune_dir,
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
        return evaluate_video_model(
            clips_root=dataset,
            model_path=checkpoint,
            output_path=output_dir,
            num_frames=spec.options.get("num_frames", 16),
            split=split,
            input_size=spec.input_size,
            backbone=self.backbone,
        )

    def infer_option_keys(self) -> set[str]:
        """:class:`VideoClassifier` fields other than the checkpoint."""
        return {"num_frames", "stride", "batch_size"}

    def check_spec(self, spec, where) -> None:
        """VideoMAE's position embeddings are fixed to its 224 px checkpoint."""
        if spec.input_size != 224:
            raise ValueError(f"{where} videomae needs options.input_size = 224")

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """A :class:`VideoClassifier`; ``infer_options`` are its fields.

        ``num_frames`` follows training (``spec.options``) unless set there.
        """
        trained = {k: v for k, v in spec.options.items() if k == "num_frames"}
        return VideoClassifier(
            model_path=str(checkpoint),
            input_type=spec.input,
            input_size=spec.input_size,
            backbone=self.backbone,
            **(trained | spec.infer_options),
        )

    def prepared_selection(self, layout, dataset_id) -> Path:
        return layout.videomae_clips(dataset_id)

    @staticmethod
    def _recrop(spec) -> bool:
        """Bird crops of a size other than the Classify videos' 224 px."""
        return spec.input == "bbox_crop" and spec.input_size != 224

    def dataset_export(self, media, spec):
        cage_media = spec.input == "cage" or self._recrop(spec)
        return media.detect_export if cage_media else media.classify_export

    def build_dataset(
        self, selection, ground_truth, spec, media, out_dir, guidance_path
    ) -> None:
        """Clips, ``{split}/{label}/{clip}/``, one per selected clip.

        Bird-crop clips by default: crop frame ``k`` is the cage's ``k``-th
        boxed canonical frame (ADR 0002). With ``input = "cage"``, whole cage
        frames squared to ``input_size`` px, cut from the cage videos (frame ``k`` is
        canonical ``k - 1``). Bird crops other than 224 px are recropped
        from the cage videos at ``input_size`` around the smoothed
        ground-truth box centers, as inference crops them (ADR 0011).
        Clips shorter than ``num_frames`` widen into neighbouring frames.

        Raises:
            ValueError: If a bird-crop clip covers a frame with no box, or
                boxed frames that aren't contiguous in the crop.
            RuntimeError: If any selected clip was not written.
        """
        if spec.input == "cage":
            ranges = self._cage_ranges(selection)
        else:
            ranges = self._crop_ranges(selection, ground_truth)
        selection.select(
            video_path=pl.col("camera_id") + "/" + pl.col("video_id") + ".mp4",
            split="split",
        ).unique().write_parquet(guidance_path)
        num_frames = spec.options["num_frames"]
        stream_annotations_to_mae(
            path_to_videos=Path(),  # unused: media come from the export
            path_to_annotations=self.dataset_export(media, spec),
            path_to_guidance=guidance_path,
            path_to_output=out_dir,
            num_frames=num_frames,
            # Each selected clip is at most num_frames long: one clip per range
            stride=num_frames,
            processes=media.processes,
            ranges_by_cage=ranges,
            square_size=spec.input_size if spec.input == "cage" else None,
            crop_centers_by_cage=ground_truth_crop_centers(ground_truth, ranges)
            if self._recrop(spec)
            else None,
            crop_size=spec.input_size,
        )
        split_of = dict(selection.select("camera_id", "split").unique().iter_rows())
        missing = [
            f"{key}/{i}"
            for key, key_ranges in ranges.items()
            for i, (label, _, _) in enumerate(key_ranges)
            if not (
                out_dir
                / split_of[key[0]]
                / label_dir_name(label)
                / f"{key[0]}_{key[1]}_{key[2]}_s{i}_clip000"
            ).is_dir()
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {selection.height} selected clips were not "
                f"written to {out_dir}, e.g. {missing[:10]}"
            )

    @staticmethod
    def _cage_ranges(
        selection,
    ) -> dict[tuple[str, str, str], list[tuple[str, int, int]]]:
        """Each clip as 1-indexed cage-video frames, by cage."""
        ranges: dict[tuple[str, str, str], list[tuple[str, int, int]]] = {}
        for row in selection.sort(
            "camera_id", "video_id", "cage_id", "frame_begin"
        ).iter_rows(named=True):
            key = (row["camera_id"], row["video_id"], row["cage_id"])
            ranges.setdefault(key, []).append(
                (
                    row["label"],
                    canonical_to_ls(row["frame_begin"]),
                    canonical_to_ls(row["frame_end"]),
                )
            )
        return ranges

    @staticmethod
    def _crop_ranges(
        selection, ground_truth
    ) -> dict[tuple[str, str, str], list[tuple[str, int, int]]]:
        """Each clip as 1-indexed bird-crop frames, by cage."""
        crop_index: dict[tuple[str, str, str], dict[int, int]] = {}
        boxed = (
            ground_truth.filter(pl.col("bbox_left").is_not_null())
            .sort("camera_id", "video_id", "cage_id", "frame")
            .group_by("camera_id", "video_id", "cage_id", maintain_order=True)
            .agg("frame")
        )
        for camera_id, video_id, cage_id, frames in boxed.iter_rows():
            crop_index[(camera_id, video_id, cage_id)] = {
                f: i for i, f in enumerate(frames)
            }
        ranges: dict[tuple[str, str, str], list[tuple[str, int, int]]] = {}
        for row in selection.sort(
            "camera_id", "video_id", "cage_id", "frame_begin"
        ).iter_rows(named=True):
            key = (row["camera_id"], row["video_id"], row["cage_id"])
            index = crop_index.get(key, {})
            begin, end = index.get(row["frame_begin"]), index.get(row["frame_end"])
            if (
                begin is None
                or end is None
                or (end - begin != row["frame_end"] - row["frame_begin"])
            ):
                raise ValueError(f"Clip {row} isn't a contiguous boxed range")
            ranges.setdefault(key, []).append((row["label"], begin + 1, end + 1))
        return ranges


class RfdetrTrainer(YoloTrainer):
    """RF-DETR on the YOLO kind's dataset, scored on val presence F1 (ADR 0010).

    ``spec.pretrained`` is an ``rfdetr`` model class (e.g. ``"RFDETRSmall"``);
    ``spec.options`` are its ``model.train`` arguments. Not tunable.
    """

    def checkpoint_path(self, run_dir: Path) -> Path:
        return run_dir / "checkpoint_best_total.pth"

    def train(self, spec, dataset, run_dir, hyperparameters, context) -> TrainResult:
        result = train_rfdetr(
            dataset_root=dataset,
            run_dir=run_dir,
            pretrained=spec.pretrained,
            epochs=spec.epochs,
            device=context.device,
            train_options=spec.options | hyperparameters,
            select_conf=context.select_conf,
            iou_threshold=context.iou_threshold,
        )
        return TrainResult(
            checkpoint=result.pop("checkpoint"),
            hyperparameters=dict(hyperparameters),
            metrics=result,
        )

    def tune(self, spec, dataset, run_dir, context) -> TrainResult:
        logger.warning("rfdetr has no tuning; training once with fixed options")
        return self.train(spec, dataset, run_dir, {}, context)

    def evaluate(
        self, checkpoint, spec, dataset, split, output_dir, context
    ) -> dict[str, Any]:
        return rfdetr_presence_metrics(
            dataset,
            checkpoint,
            split=split,
            conf=context.select_conf,
            iou_threshold=context.iou_threshold,
            device=context.device,
        )

    def infer_option_keys(self) -> set[str]:
        """:class:`RfdetrDetector` fields; ``tracker_path`` is required."""
        return {"tracker_path", "threshold", "batch_size"}

    def load_for_inference(self, checkpoint: Path, spec: "ModelSpec") -> Any:
        """An :class:`RfdetrDetector`; ``infer_options`` are its fields."""
        if "tracker_path" not in spec.infer_options:
            raise ValueError("rfdetr needs infer_options.tracker_path")
        return RfdetrDetector(model_path=checkpoint, **spec.infer_options)


class VJEPA2Trainer(VideoMAETrainer):
    """V-JEPA 2 clip classifier: VideoMAE's training with another backbone (ADR 0011).

    Any ``input_size`` that fits its 16 px patches; a frozen encoder still
    trains the attentive pooler and the classifier.
    """

    backbone = "vjepa2"

    def check_spec(self, spec, where) -> None:
        num_frames = spec.options.get("num_frames", 16)
        if spec.input_size % 16 or num_frames % 2:
            raise ValueError(
                f"{where} vjepa2 needs options.input_size a multiple of 16 and"
                f" an even options.num_frames, not {spec.input_size}, {num_frames}"
            )
        if not (
            spec.pretrained.startswith("facebook/vjepa2")
            or Path(spec.pretrained).is_absolute()
        ):
            raise ValueError(
                f"{where} vjepa2 pretrained must be a facebook/vjepa2-* model"
                f" or an absolute path, not {spec.pretrained!r}"
            )


def _trainers() -> dict[str, type]:
    """The registry: ``ModelSpec.kind`` to trainer class."""
    return {
        "yolo": YoloTrainer,
        "rfdetr": RfdetrTrainer,
        "videomae": VideoMAETrainer,
        "vjepa2": VJEPA2Trainer,
    }


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
