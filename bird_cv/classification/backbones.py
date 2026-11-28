"""What differs between the Hugging Face video classifiers a classifier kind runs.

Training, tuning, evaluation and windowed inference are shared; each
:class:`Backbone` names only its model class, processor, head and size
attribute. Everything tunable (checkpoint, ``num_frames``, ``input_size``)
comes from ``[models.classifier]``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from transformers import (
    AutoImageProcessor,
    AutoVideoProcessor,
    VideoMAEForVideoClassification,
    VJEPA2ForVideoClassification,
)


@dataclass(frozen=True)
class Backbone:
    """One Hugging Face video-classification architecture.

    Attributes:
        model_class: Its ``*ForVideoClassification`` class.
        processor: Loads its processor for ``(path, num_frames, input_size)``.
        head: Top-level modules that still train with a frozen encoder.
        size_key: Model config attribute holding the input side in px.
    """

    model_class: Any
    processor: Callable[[str | Path, int, int], Any]
    head: tuple[str, ...]
    size_key: str

    def pretrained(
        self, path: str | Path, num_frames: int, input_size: int, **kwargs: Any
    ) -> Any:
        """Weights to fine-tune, their config's input side set to ``input_size``."""
        model = self.model_class.from_pretrained(
            path, **{self.size_key: input_size}, **kwargs
        )
        self.check(model.config, num_frames, input_size)
        return model

    def fine_tuned(self, path: str | Path, num_frames: int, input_size: int) -> Any:
        """A fine-tuned checkpoint, checked against the run's settings."""
        model = self.model_class.from_pretrained(path)
        self.check(model.config, num_frames, input_size)
        return model

    def check(self, config: Any, num_frames: int, input_size: int) -> None:
        """Raise if ``num_frames`` or ``input_size`` don't fit the model config."""
        trained = getattr(config, self.size_key)
        if trained != input_size:
            raise ValueError(
                f"Model takes {trained} px input, options.input_size is {input_size}"
            )
        if input_size % config.patch_size or num_frames % config.tubelet_size:
            raise ValueError(
                f"input_size {input_size} and num_frames {num_frames} must be "
                f"multiples of the patch size {config.patch_size} and tubelet "
                f"size {config.tubelet_size}"
            )


def _videomae_processor(path: str | Path, num_frames: int, input_size: int) -> Any:
    return AutoImageProcessor.from_pretrained(str(path), num_frames=num_frames)


def _vjepa2_processor(path: str | Path, num_frames: int, input_size: int) -> Any:
    # Frames already arrive at input_size: normalize only, never resize
    return AutoVideoProcessor.from_pretrained(
        str(path), crop_size=input_size, do_resize=False, do_center_crop=False
    )


def get_backbone(kind: str) -> Backbone:
    """The backbone of a classifier ``kind`` (``"videomae"`` or ``"vjepa2"``)."""
    return {
        # VideoMAE's head is mean pooling, fc_norm and a linear classifier;
        # only the linear layer is new
        "videomae": Backbone(
            VideoMAEForVideoClassification,
            _videomae_processor,
            ("classifier",),
            "image_size",
        ),
        # V-JEPA 2's attentive pooler is new, so it trains with the classifier;
        # its RoPE grid follows config.crop_size, hence the override
        "vjepa2": Backbone(
            VJEPA2ForVideoClassification,
            _vjepa2_processor,
            ("pooler", "classifier"),
            "crop_size",
        ),
    }[kind]


def pixel_values(processor: Any, frames: list[np.ndarray]) -> torch.Tensor:
    """One clip's pixel tensor: the processor's only output, batch dim dropped.

    VideoMAE's processor names it ``pixel_values``, V-JEPA 2's
    ``pixel_values_videos``; both models take it as the first argument.
    """
    (values,) = processor(frames, return_tensors="pt").values()
    return values.squeeze(0)
