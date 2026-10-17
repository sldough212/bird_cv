"""Inference models chosen by config, never by pipeline code.

The detector and classifier come from the ``[train.detector]`` and
``[train.classifier]`` :class:`~bird_cv.pipelines.config.ModelSpec` tables
through :func:`bird_cv.training.registry.get_trainer` (one ``kind`` mapping
per role, shared by training and inference). The cage segmenter, which is
never trained, has its own mapping here (:func:`_segmenters`), picked by
``[infer.segmenter] kind``. Importing this module imports no model library.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from bird_cv.inference.protocols import (
        CageSegmenter,
        FrameClassifier,
        StreamingDetector,
    )
    from bird_cv.pipelines.config import BirdCVConfig, ModelSpec, SegmenterSpec


@dataclass(frozen=True)
class SegmenterContext:
    """Run-wide settings a segmenter may need beyond its options.

    Attributes:
        segmentation_configs: ``path_to_segmentation_configs``.
        sam_model_id: ``sam_model_id``.
    """

    segmentation_configs: Path
    sam_model_id: str


class Segmenters(Protocol):
    """Builds one :class:`CageSegmenter` per camera."""

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        """Fail before any work if a camera can't be segmented."""
        ...

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        """The segmenter for one camera's videos."""
        ...


class Sam2Segmenters:
    """SAM2 prompted per camera from ``{segmentation_configs}/configs/``.

    One predictor is built on first use and shared by every camera.
    ``options`` are :class:`~bird_cv.inference.cages.Sam2CageSegmenter`
    keyword arguments (e.g. ``iou_threshold``).
    """

    def __init__(self, context: SegmenterContext, **options: Any) -> None:
        self.context = context
        self.options = options
        self._predictor: Any = None
        self._device: str | None = None

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        from bird_cv.inference.cages import check_sam2_configs

        check_sam2_configs(self.context.segmentation_configs, camera_ids)

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        import torch
        from sam2.build_sam import build_sam2_video_predictor_hf

        from bird_cv.inference.cages import (
            Sam2CageSegmenter,
            sam2_config_path,
            sam2_reference_frame,
        )

        if self._predictor is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._predictor = build_sam2_video_predictor_hf(
                self.context.sam_model_id, device=self._device
            )
        configs = self.context.segmentation_configs
        return Sam2CageSegmenter(
            predictor=self._predictor,
            sam_model_id=self.context.sam_model_id,
            config_path=sam2_config_path(configs, camera_id),
            reference_frame_path=sam2_reference_frame(configs, camera_id),
            device=self._device or "cuda",
            **self.options,
        )


class WholeFrameSegmenters:
    """Every frame is one cage (videos that already show a single cage)."""

    def __init__(self, context: SegmenterContext) -> None:
        self.context = context

    def check_cameras(self, camera_ids: Iterable[str]) -> None:
        return None

    def for_camera(self, camera_id: str) -> "CageSegmenter":
        from bird_cv.inference.cages import WholeFrameSegmenter

        return WholeFrameSegmenter()


def _segmenters() -> dict[str, type]:
    """The segmenter registry: ``[infer.segmenter] kind`` to class."""
    return {"sam2": Sam2Segmenters, "whole_frame": WholeFrameSegmenters}


def segmenter_kinds() -> list[str]:
    """Every registered segmenter ``kind``, sorted."""
    return sorted(_segmenters())


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
        detector: The ``[train.detector]`` spec.
        classifier: The ``[train.classifier]`` spec.
        checkpoints: Weights per role (``detector``, ``classifier``).
        context: Settings the segmenter reads.
    """

    segmenter: "SegmenterSpec"
    detector: "ModelSpec"
    classifier: "ModelSpec"
    checkpoints: Mapping[str, Path]
    context: SegmenterContext

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
        """The per-camera segmenter factory (loads nothing until used)."""
        return get_segmenters(self.segmenter, self.context)

    def load(self) -> tuple["StreamingDetector", "FrameClassifier"]:
        """Build the detector and classifier (loads weights)."""
        from bird_cv.training.registry import get_trainer

        def load_role(role: str, spec: "ModelSpec") -> Any:
            return get_trainer(spec.kind).load_for_inference(
                self.checkpoints[role], spec
            )

        return (
            load_role("detector", self.detector),
            load_role("classifier", self.classifier),
        )


def inference_models(
    cfg: "BirdCVConfig", train_run_id: str | None = None, fold: int | None = None
) -> InferenceModels:
    """The models ``cfg`` picks, with checkpoints from a training run or config.

    Args:
        cfg: Pipeline config.
        train_run_id: Use the checkpoints this training run produced
            (:meth:`Layout.train_run` plus the kind's
            :meth:`~bird_cv.training.registry.Trainer.checkpoint_path`).
            If None, each role's ``checkpoint`` from config.
        fold: The CV fold of ``train_run_id``, if any.

    Returns:
        The resolved, unloaded models.

    Raises:
        ValueError: If a role has neither a training run nor a ``checkpoint``.
    """
    from bird_cv.core.layout import Layout
    from bird_cv.training.registry import get_trainer

    layout = Layout(cfg.path_to_base_output)
    specs = {"detector": cfg.train.detector, "classifier": cfg.train.classifier}
    checkpoints: dict[str, Path] = {}
    for role, spec in specs.items():
        if train_run_id is not None:
            run_dir = layout.train_run(role, train_run_id, fold)
            checkpoints[role] = get_trainer(spec.kind).checkpoint_path(run_dir)
        elif spec.checkpoint is not None:
            checkpoints[role] = spec.checkpoint
        else:
            raise ValueError(
                f"No {role} checkpoint: pass a training run id or set "
                f"[train.{role}] checkpoint"
            )
    return InferenceModels(
        segmenter=cfg.infer.segmenter,
        detector=specs["detector"],
        classifier=specs["classifier"],
        checkpoints=checkpoints,
        context=SegmenterContext(
            segmentation_configs=cfg.path_to_segmentation_configs,
            sam_model_id=cfg.sam_model_id,
        ),
    )
