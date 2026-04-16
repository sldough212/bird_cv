"""Evaluate a fine-tuned VideoMAE model on the test split."""

from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from bird_cv.classification.train_video_model import BehaviorClipDataset


def evaluate_video_model(
    clips_root: Path,
    model_path: Path,
    output_path: Path,
    num_frames: int = 16,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Run inference on the test split and return classification metrics.

    Loads the fine-tuned model from ``model_path``, runs inference on all
    clips under ``clips_root/test/``, and writes a confusion matrix PNG and
    classification report txt to ``output_path``.

    Args:
        clips_root: Root directory containing the ``test/`` split produced
            by :func:`extract_behavior_clips`.
        model_path: Path to the saved model directory (output of
            ``model.save_pretrained``).
        output_path: Directory where the confusion matrix and report are saved.
        num_frames: Frames per clip — must match training. Defaults to 16.
        batch_size: Dataloader batch size. Defaults to 8.
        device: Device string. Defaults to ``"cuda"``.

    Returns:
        dict with keys ``"report"`` (sklearn classification report dict) and
        ``"confusion_matrix"`` (numpy array, rows=true, cols=predicted).
    """
    output_path.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_path, num_frames=num_frames)
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model = model.to(dev)
    model.eval()

    test_ds = BehaviorClipDataset(clips_root / "test", processor, num_frames)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    label_names = [
        d.name for d in sorted((clips_root / "test").iterdir()) if d.is_dir()
    ]

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(dev)
            labels = batch["labels"].to(dev)
            outputs = model(pixel_values=pixel_values)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds, target_names=label_names, output_dict=True
    )
    report_str = classification_report(all_labels, all_preds, target_names=label_names)
    print(report_str)
    (output_path / "classification_report.txt").write_text(report_str)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(
        figsize=(max(6, len(label_names)), max(5, len(label_names) - 1))
    )
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names).plot(
        ax=ax, colorbar=False, xticks_rotation=45
    )
    ax.set_title("Behavior Classification — Test Set")
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {"report": report, "confusion_matrix": cm}
