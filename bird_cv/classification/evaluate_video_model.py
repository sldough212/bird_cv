"""Evaluate a fine-tuned video classifier on one split of its clip dataset."""

from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader
from bird_cv.classification.backbones import get_backbone
from bird_cv.classification.train_video_model import (
    BehaviorClipDataset,
    predict_clip_indices,
)
from bird_cv.core.layout import EvalRun
from bird_cv.evaluation.metrics import classification_metrics


def evaluate_video_model(
    clips_root: Path,
    model_path: Path,
    output_path: Path,
    num_frames: int = 16,
    batch_size: int = 8,
    device: str = "cuda",
    *,
    split: str = "test",
    input_size: int = 224,
    backbone: str = "videomae",
) -> dict:
    """Score a model on ``clips_root/{split}/`` with window macro-F1.

    Clips are indexed by the model's own ``label2id``, so a split missing a
    class is scored correctly; clips with labels outside it are skipped and
    counted. Writes per-class scores and the confusion matrix (rows true)
    to ``output_path`` as ``per_class.parquet`` and ``confusion.parquet``.

    Args:
        clips_root: Root directory containing the split produced by
            :func:`~bird_cv.preprocessing.annotations_to_mae.stream_annotations_to_mae`.
        model_path: Path to the saved model directory (output of
            ``model.save_pretrained``).
        output_path: Directory the per-class scores and confusion matrix go to.
        num_frames: Frames per clip — must match training. Defaults to 16.
        batch_size: Dataloader batch size. Defaults to 8.
        device: Device string. Defaults to ``"cuda"``.
        split: Split subdirectory of ``clips_root`` to evaluate. Defaults to
            ``"test"``.
        input_size: Side in px of the clip frames; must match the checkpoint.
        backbone: Classifier kind, see
            :func:`~bird_cv.classification.backbones.get_backbone`.

    Returns:
        ``macro_f1`` (over the classes present in truth or predictions, see
        :func:`~bird_cv.evaluation.metrics.classification_metrics`) and
        ``n_skipped_out_of_vocabulary``.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    bb = get_backbone(backbone)
    model = bb.fine_tuned(model_path, num_frames, input_size).to(dev)
    processor = bb.processor(model_path, num_frames, input_size)
    model.eval()

    label_to_idx = dict(model.config.label2id)
    test_ds = BehaviorClipDataset(
        clips_root / split, processor, num_frames, label_to_idx=label_to_idx
    )
    classes = [name for name, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    all_labels, all_preds = predict_clip_indices(model, test_loader, dev)

    metrics = classification_metrics(
        [classes[i] for i in all_labels], [classes[i] for i in all_preds], classes
    )
    out = EvalRun(output_path)
    metrics.per_class.write_parquet(out.per_class)
    pl.DataFrame(
        {"label": classes} | {c: metrics.confusion[:, j] for j, c in enumerate(classes)}
    ).write_parquet(out.confusion)
    print(metrics.per_class)
    print(f"Window macro-F1 ({split}): {metrics.macro_f1:.4f}")
    return {
        "macro_f1": metrics.macro_f1,
        "n_skipped_out_of_vocabulary": test_ds.n_skipped_clips,
    }
