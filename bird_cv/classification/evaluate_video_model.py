"""Evaluate a fine-tuned VideoMAE model on the test split."""

from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from bird_cv.classification.train_video_model import (
    BehaviorClipDataset,
    predict_clip_indices,
)
from bird_cv.evaluation.metrics import window_macro_f1 as compute_window_macro_f1


def evaluate_video_model(
    clips_root: Path,
    model_path: Path,
    output_path: Path,
    num_frames: int = 16,
    batch_size: int = 8,
    device: str = "cuda",
    *,
    split: str = "test",
    window_macro_f1: bool = False,
) -> dict:
    """Run inference on the test split and return classification metrics.

    Loads the fine-tuned model from ``model_path``, runs inference on all
    clips under ``clips_root/{split}/`` (``test`` by default), and writes a
    confusion matrix PNG and classification report txt to ``output_path``.

    Args:
        clips_root: Root directory containing the ``test/`` split produced
            by :func:`extract_behavior_clips`.
        model_path: Path to the saved model directory (output of
            ``model.save_pretrained``).
        output_path: Directory where the confusion matrix and report are saved.
        num_frames: Frames per clip — must match training. Defaults to 16.
        batch_size: Dataloader batch size. Defaults to 8.
        device: Device string. Defaults to ``"cuda"``.
        split: Split subdirectory of ``clips_root`` to evaluate. Defaults to
            ``"test"``.
        window_macro_f1: If ``True``, restrict the report and confusion
            matrix to classes present in truth or predictions, and add the
            single headline number
            ``"macro_f1"`` from
            :func:`bird_cv.evaluation.metrics.window_macro_f1` (equal to the
            report's macro avg F1). Defaults to ``False`` (one row per
            model class). Either way clips are indexed by the model's own
            ``label2id``, so a split missing a class is scored correctly;
            clips with labels outside it are skipped and counted.

    Returns:
        dict with keys ``"report"`` (sklearn classification report dict) and
        ``"confusion_matrix"`` (numpy array, rows=true, cols=predicted), plus
        ``"macro_f1"`` and ``"n_skipped_out_of_vocabulary"`` when
        ``window_macro_f1`` is set.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_path, num_frames=num_frames)
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model = model.to(dev)
    model.eval()

    # Index clips by the model's own vocabulary so a split missing a class
    # is scored against the right indices; out-of-vocabulary clips are
    # skipped and counted.
    label_to_idx = dict(model.config.label2id)
    test_ds = BehaviorClipDataset(
        clips_root / split, processor, num_frames, label_to_idx=label_to_idx
    )
    label_names = [
        name for name, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])
    ]
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    all_labels, all_preds = predict_clip_indices(model, test_loader, dev)

    # Default: one report row per model class, including absent ones.
    report_kwargs: dict = {"labels": list(range(len(label_names)))}
    report_header = ""
    if window_macro_f1:
        # Report exactly the classes present in truth or predictions, the
        # set metrics.window_macro_f1 averages over, so the report's
        # "macro avg" F1 equals results["macro_f1"].
        vocabulary = label_names
        present = sorted(set(all_labels) | set(all_preds))
        label_names = [vocabulary[i] for i in present]
        report_kwargs = {"labels": present}
        report_header = (
            f"Window metrics, {split} split: classes present in truth or"
            " predictions; macro avg F1 is the window macro-F1.\n"
            f"Clips skipped (label outside the model vocabulary):"
            f" {test_ds.n_skipped_clips}\n\n"
        )
    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        output_dict=True,
        **report_kwargs,
    )
    report_str = report_header + classification_report(
        all_labels, all_preds, target_names=label_names, **report_kwargs
    )
    print(report_str)
    (output_path / "classification_report.txt").write_text(report_str)

    cm = confusion_matrix(all_labels, all_preds, **report_kwargs)
    fig, ax = plt.subplots(
        figsize=(max(6, len(label_names)), max(5, len(label_names) - 1))
    )
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names).plot(
        ax=ax, colorbar=False, xticks_rotation=45
    )
    ax.set_title(f"Behavior Classification — {split.capitalize()} Set")
    fig.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    results = {"report": report, "confusion_matrix": cm}
    if window_macro_f1:
        results["macro_f1"] = compute_window_macro_f1(
            [vocabulary[i] for i in all_labels],
            [vocabulary[i] for i in all_preds],
            classes=vocabulary,
        )
        results["n_skipped_out_of_vocabulary"] = test_ds.n_skipped_clips
        print(f"Window macro-F1 ({split}): {results['macro_f1']:.4f}")
    return results
