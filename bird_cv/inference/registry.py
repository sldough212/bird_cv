"""Inference models chosen by config, never by pipeline code.

The detector and classifier come from the ``[models.detector]`` and
``[models.classifier]`` :class:`~bird_cv.pipelines.config.ModelSpec` tables
through :func:`bird_cv.training.registry.get_trainer` (one ``kind`` mapping
per role, shared by training and inference). The cage segmenter has its
own mapping here (:func:`_segmenters`), picked by ``[infer.segmenter]
kind``; one that loads a trained model names its role (``trained_role``),
whose checkpoint is resolved like the detector's.
"""

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from sam2.build_sam import build_sam2_video_predictor_hf

from bird_cv.core.layout import Layout
from bird_cv.core.manifest import read_manifest
from bird_cv.inference.cages import (
    Sam2CageSegmenter,
    WholeFrameSegmenter,
    YoloCageSegmenter,
    check_sam2_configs,
)
from bird_cv.segmentation.segment import sam2_config_path, sam2_reference_frame
from bird_cv.inference.protocols import (
    CageSegmenter,
    FrameClassifier,
    StreamingDetector,
)
from bird_cv.training.registry import get_trainer

if TYPE_CHECKING:
    from bird_cv.pipelines.config import BirdCVConfig, ModelSpec, SegmenterSpec


@dataclass(frozen=True)
class SegmenterContext:
    """Run-wide settings a segmenter may need beyond its options.

    Attributes:
        segmentation_configs: ``path_to_segmentation_configs``.
        sam_model_id: ``sam_model_id``.
        trained: The ``trained_role`` spec, if the segmenter has one.
        checkpoint: That role's resolved checkpoint.
    """

    segmentation_configs: Path
    sam_model_id: str
    trained: "ModelSpec | None" = None
    checkpoint: Path | None = None


class Segmenters(Protocol):
    """Builds one :class:`CageSegmenter` per camera."""

    @classmethod
    def option_keys(cls) -> set[str]:
        """``[infer.segmenter] options`` this kind accepts (checked at config load)."""
        ...

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        """Fail before any work if a camera can't be segmented."""
        ...

    def load(self) -> None:
        """Load model weights; a failure here stops the task, not one video."""
        ...

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        """The segmenter for one camera's videos (after :meth:`load`)."""
        ...


class Sam2Segmenters:
    """SAM2 prompted per camera from ``{segmentation_configs}/configs/``.

    :meth:`load` builds one predictor, shared by every camera. ``options``
    are :class:`~bird_cv.inference.cages.Sam2CageSegmenter` keyword
    arguments (:meth:`option_keys`), checked against its signature here.
    """

    def __init__(self, context: SegmenterContext, **options: Any) -> None:
        unknown = set(options) - self.option_keys()
        if unknown:
            raise ValueError(f"Unknown sam2 segmenter options: {sorted(unknown)}")
        inspect.signature(Sam2CageSegmenter).bind_partial(**options)
        self.context = context
        self.options = options
        self._predictor: Any = None
        self._device = "cuda"

    @classmethod
    def option_keys(cls) -> set[str]:
        return {"iou_threshold"}

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        check_sam2_configs(self.context.segmentation_configs, camera_ids)

    def load(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._predictor = build_sam2_video_predictor_hf(
            self.context.sam_model_id, device=self._device
        )

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        if self._predictor is None:
            raise RuntimeError("Call Sam2Segmenters.load() before for_camera()")
        configs = self.context.segmentation_configs
        return Sam2CageSegmenter(
            predictor=self._predictor,
            sam_model_id=self.context.sam_model_id,
            config_path=sam2_config_path(configs, camera_id),
            reference_frame_path=sam2_reference_frame(configs, camera_id),
            device=self._device,
            **self.options,
        )


class WholeFrameSegmenters:
    """Every frame is one cage (videos that already show a single cage)."""

    def __init__(self, context: SegmenterContext) -> None:
        self.context = context

    @classmethod
    def option_keys(cls) -> set[str]:
        return set()

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        return None

    def load(self) -> None:
        return None

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        return WholeFrameSegmenter()


class YoloCageSegmenters:
    """The trained ``[models.cage_detector]`` model, no prompt configs (ADR 0007).

    ``options`` are :class:`~bird_cv.inference.cages.YoloCageSegmenter`
    arguments; ``expected_cages`` is an int for every camera or a
    ``{camera_id: int}`` table (cameras not in it: automatic).
    """

    trained_role = "cage_detector"

    def __init__(
        self,
        context: SegmenterContext,
        conf: float = 0.5,
        sample_frames: int = 5,
        expected_cages: int | dict[str, int] | None = None,
    ) -> None:
        self.context = context
        self.conf = conf
        self.sample_frames = sample_frames
        self.expected_cages = expected_cages or {}
        self._model: Any = None

    @classmethod
    def option_keys(cls) -> set[str]:
        return {"conf", "sample_frames", "expected_cages"}

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        if self.context.trained is None or self.context.checkpoint is None:
            raise ValueError("yolo_cages needs a [models.cage_detector] model")

    def load(self) -> None:
        assert self.context.trained and self.context.checkpoint  # check_cameras
        trainer = get_trainer(self.context.trained.kind)
        self._model = trainer.load_for_inference(
            self.context.checkpoint, self.context.trained
        )

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        if self._model is None:
            raise RuntimeError("Call YoloCageSegmenters.load() before for_camera()")
        expected = self.expected_cages
        return YoloCageSegmenter(
            self._model,
            expected_cages=expected
            if isinstance(expected, int)
            else expected.get(camera_id),
            conf=self.conf,
            sample_frames=self.sample_frames,
        )


def _segmenters() -> dict[str, Any]:
    """The segmenter registry: ``[infer.segmenter] kind`` to class."""
    return {
        "sam2": Sam2Segmenters,
        "yolo_cages": YoloCageSegmenters,
        "whole_frame": WholeFrameSegmenters,
    }


def segmenter_kinds() -> list[str]:
    """Every registered segmenter ``kind``, sorted."""
    return sorted(_segmenters())


def segmenter_option_keys(kind: str) -> set[str]:
    """The options a registered segmenter ``kind`` accepts."""
    return _segmenters()[kind].option_keys()


def get_segmenters(spec: "SegmenterSpec", context: SegmenterContext) -> Segmenters:
    """The per-camera segmenter factory for ``spec``.

    Raises:
        ValueError: If no segmenter is registered for ``spec.kind``.
    """
    registry = _segmenters()
    if spec.kind not in registry:
        raise ValueError(
            f"No segmenter for kind {spec.kind!r}; known: {segmenter_kinds()}"
        )
    return registry[spec.kind](context, **spec.options)


@dataclass(frozen=True)
class InferenceModels:
    """Which models an inference run uses, resolved from config but not loaded.

    Attributes:
        segmenter: The ``[infer.segmenter]`` spec.
        detector: The ``[models.detector]`` spec.
        classifier: The ``[models.classifier]`` spec.
        checkpoints: Weights per role (``detector``, ``classifier``).
        context: Settings the segmenter reads.
        train_run_id: The training run the checkpoints came from, if any.
        dataset: The dataset (``{"id", "key"}``) that training run recorded,
            if any.
    """

    segmenter: "SegmenterSpec"
    detector: "ModelSpec"
    classifier: "ModelSpec"
    checkpoints: Mapping[str, Path]
    context: SegmenterContext
    train_run_id: str | None = None
    dataset: dict[str, Any] | None = None

    def kinds(self) -> dict[str, str]:
        """Each role's ``kind``, for manifests."""
        return {
            "segmenter": self.segmenter.kind,
            "detector": self.detector.kind,
            "classifier": self.classifier.kind,
        }

    def inputs(self) -> dict[str, str]:
        """Upstream files the models read, for manifests."""
        return {role: str(path) for role, path in self.checkpoints.items()} | {
            "segmentation_configs": str(self.context.segmentation_configs)
        }

    def segmenters(self) -> Segmenters:
        """The per-camera segmenter factory (loads nothing until ``load()``)."""
        return get_segmenters(self.segmenter, self.context)

    def load(self) -> tuple["StreamingDetector", "FrameClassifier"]:
        """Build the detector and classifier (loads weights)."""

        def load_role(role: str, spec: "ModelSpec") -> Any:
            return get_trainer(spec.kind).load_for_inference(
                self.checkpoints[role], spec
            )

        return (
            load_role("detector", self.detector),
            load_role("classifier", self.classifier),
        )


def inference_models(
    cfg: "BirdCVConfig", train_run_id: str | None = None
) -> InferenceModels:
    """The models ``cfg`` picks, with checkpoints from a training run or config.

    Args:
        cfg: Pipeline config.
        train_run_id: Use the checkpoints this training run produced
            (:meth:`Layout.train_run` plus the kind's
            :meth:`~bird_cv.training.registry.Trainer.checkpoint_path`).
            If None, each role's ``checkpoint`` from config.

    Returns:
        The resolved, unloaded models.

    The segmenter's ``trained_role`` (if any) is resolved the same way.

    Raises:
        ValueError: If a role has neither a training run nor a ``checkpoint``,
            or the segmenter's role isn't configured.
        FileNotFoundError: If a role's checkpoint doesn't exist.
        ValueError: If the training runs of the roles used different
            datasets.
    """
    layout = Layout(cfg.path_to_base_output)
    specs = {"detector": cfg.models.detector, "classifier": cfg.models.classifier}
    trained_role = getattr(
        _segmenters()[cfg.infer.segmenter.kind], "trained_role", None
    )
    if trained_role is not None:
        trained = cfg.models.roles().get(trained_role)
        if trained is None:
            raise ValueError(
                f"[infer.segmenter] kind {cfg.infer.segmenter.kind!r} needs "
                f"[models.{trained_role}]"
            )
        specs[trained_role] = trained
    checkpoints: dict[str, Path] = {}
    datasets: dict[str, dict[str, Any] | None] = {}
    for role, spec in specs.items():
        if train_run_id is not None:
            run_dir = layout.train_run(role, train_run_id)
            checkpoints[role] = get_trainer(spec.kind).checkpoint_path(run_dir)
            if (run_dir / "manifest.json").exists():
                datasets[role] = read_manifest(run_dir).config.get("dataset")
        elif spec.checkpoint is not None:
            checkpoints[role] = spec.checkpoint
        else:
            raise ValueError(
                f"No {role} checkpoint: pass a training run id or set "
                f"[models.{role}] checkpoint"
            )
        if not checkpoints[role].exists():
            source = (
                f"training run {train_run_id!r}"
                if train_run_id is not None
                else f"[models.{role}] checkpoint"
            )
            raise FileNotFoundError(
                f"{role} checkpoint {checkpoints[role]} (from {source}) does not exist"
            )
    if len({str(d and d["key"]) for d in datasets.values()}) > 1:
        raise ValueError(
            f"Training run {train_run_id!r} roles used different datasets: {datasets}"
        )
    return InferenceModels(
        segmenter=cfg.infer.segmenter,
        detector=specs["detector"],
        classifier=specs["classifier"],
        checkpoints=checkpoints,
        context=SegmenterContext(
            segmentation_configs=cfg.path_to_segmentation_configs,
            sam_model_id=cfg.sam_model_id,
            trained=specs.get(trained_role) if trained_role else None,
            checkpoint=checkpoints.get(trained_role) if trained_role else None,
        ),
        train_run_id=train_run_id,
        dataset=next(iter(datasets.values()), None),
    )
