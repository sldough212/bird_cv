"""Train a VideoMAE-based behavior classification model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import msgspec
import numpy as np
import tomli_w
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    VideoMAEForVideoClassification,
)

from bird_cv.evaluation.metrics import window_macro_f1
from bird_cv.pipelines.config import resolve_run_dir

SelectionMetric = Literal["accuracy", "macro_f1"]


class Paths(msgspec.Struct):
    base_path: Path
    video_crop_path: Path
    model_checkpoint: Path
    output_root: str = ""
    best_checkpoint: str = ""


class Training(msgspec.Struct):
    num_frames: int = 16
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    device: str = "cuda"
    freeze_encoder: bool = True
    run_name: str = "videomae_behavior"


class ClassificationConfig(msgspec.Struct):
    paths: Paths
    training: Training


class BehaviorClipDataset(Dataset):
    """Dataset of fixed-length frame clips for behavior classification.

    Expects the directory structure produced by :func:`extract_behavior_clips`::

        clips_root/
          {split}/
            {label}/
              {track_id}_{cage_id}_clip{idx}/
                00000.jpg
                00001.jpg
                ...

    Args:
        clips_root: Path to the split directory (e.g. ``clips/train``).
        processor: HuggingFace image processor for the VideoMAE model.
        num_frames: Number of frames per clip. Must match the model's
            expected input length. Defaults to 16.
        label_to_idx: Fixed label-to-index mapping (e.g. the train split's,
            or a model's ``config.label2id``), so a split missing a class
            still uses the model's indices. Label directories not in it are
            skipped. If ``None`` (default), indices come from this split's
            own sorted label directories.
    """

    def __init__(
        self,
        clips_root: Path,
        processor: AutoImageProcessor,
        num_frames: int = 16,
        *,
        label_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames

        self.label_dirs = sorted([d for d in clips_root.iterdir() if d.is_dir()])
        if label_to_idx is None:
            self.label_to_idx = {d.name: i for i, d in enumerate(self.label_dirs)}
        else:
            skipped = [d.name for d in self.label_dirs if d.name not in label_to_idx]
            if skipped:
                print(f"  Skipping labels not in the model's vocabulary: {skipped}")
            self.label_dirs = [d for d in self.label_dirs if d.name in label_to_idx]
            self.label_to_idx = dict(label_to_idx)

        self.clips: list[tuple[Path, int]] = []
        for label_dir in self.label_dirs:
            label_idx = self.label_to_idx[label_dir.name]
            for clip_dir in sorted(label_dir.iterdir()):
                if clip_dir.is_dir():
                    self.clips.append((clip_dir, label_idx))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip_dir, label_idx = self.clips[idx]
        frame_paths = sorted(clip_dir.glob("*.jpg"))[: self.num_frames]
        frames_np = [np.array(Image.open(p)) for p in frame_paths]  # HWC uint8
        inputs = self.processor(frames_np, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }


def train_video_model(
    clips_root: Path,
    output_root: Path,
    model_checkpoint: Path,
    output_name: str = "videomae_behavior",
    num_frames: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    freeze_encoder: bool = True,
    *,
    selection_metric: SelectionMetric = "accuracy",
    weight_decay: float = 0.01,
    class_weighted_loss: bool = False,
    save_checkpoint: bool = True,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune a VideoMAE model for behavior classification.

    Loads the train and val splits from ``clips_root``, fine-tunes a
    VideoMAE classification head (and optionally the full encoder), and
    saves the best checkpoint by validation ``selection_metric``
    (accuracy by default).

    Args:
        clips_root: Root directory containing ``train/`` and ``val/``
            split subdirectories produced by :func:`extract_behavior_clips`.
        output_root: Directory where run outputs will be saved.
        output_name: Name of the run — a subdirectory with this name is
            created under ``output_root``.
        model_checkpoint: Local path to load VideoMAE weights from.
        num_frames: Frames per clip — must match what was used in
            :func:`extract_behavior_clips`. Defaults to 16.
        epochs: Number of training epochs. Defaults to 10.
        batch_size: Dataloader batch size. Defaults to 8.
        lr: Learning rate. Defaults to 1e-4.
        device: Device string passed to ``torch.device``. Defaults to
            ``"cuda"``.
        freeze_encoder: If ``True``, freeze the VideoMAE encoder and only
            train the classification head. Defaults to ``True``.
        selection_metric: Validation metric that picks the saved epoch:
            ``"accuracy"`` (default) or window ``"macro_f1"`` (ADR 0005).
            With ``"macro_f1"`` the val split is indexed by the train
            split's labels, so a val split missing a class is scored
            correctly.
        weight_decay: AdamW weight decay. Defaults to 0.01 (AdamW's own
            default).
        class_weighted_loss: If ``True``, weight the cross-entropy loss by
            inverse train-clip class frequency. Defaults to ``False``.
        save_checkpoint: If ``False``, never write a checkpoint (tuning
            trials). Defaults to ``True``.
        epoch_callback: Called after each epoch with that epoch's
            ``epoch``, ``train_loss``, ``val_accuracy`` and
            ``val_macro_f1``, e.g. to report to Ray Tune.

    Returns:
        ``selection_metric``, ``best_score``, ``best_epoch`` (1-indexed)
        and ``history`` (one dict per epoch, as passed to
        ``epoch_callback``).
    """
    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model: {model_checkpoint}")
    processor = AutoImageProcessor.from_pretrained(
        str(model_checkpoint), num_frames=num_frames
    )

    print("Building datasets...")
    train_ds = BehaviorClipDataset(clips_root / "train", processor, num_frames)
    label_to_idx = train_ds.label_to_idx
    val_ds = BehaviorClipDataset(
        clips_root / "val",
        processor,
        num_frames,
        label_to_idx=label_to_idx if selection_metric == "macro_f1" else None,
    )

    id2label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    print(f"  Labels ({num_classes}): {list(label_to_idx.keys())}")
    print(f"  Train clips: {len(train_ds)}")
    print(f"  Val clips:   {len(val_ds)}")

    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label_to_idx,
        ignore_mismatched_sizes=True,
    )

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Encoder frozen — trainable params: {trainable:,}")
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Full fine-tune — trainable params: {trainable:,}")

    model = model.to(dev)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_ds, num_classes).to(dev)
        if class_weighted_loss
        else None
    )

    print(f"\nTraining for {epochs} epochs (lr={lr}, batch_size={batch_size})\n")
    best_val_acc = 0.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [train]",
            unit="batch",
            leave=False,
        )
        for batch in train_bar:
            pixel_values = batch["pixel_values"].to(dev)
            labels = batch["labels"].to(dev)

            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = train_loss / len(train_loader)

        # Validate
        val_labels, val_preds = predict_clip_indices(
            model, val_loader, dev, desc=f"Epoch {epoch + 1}/{epochs} [val]  "
        )
        correct = sum(int(t == p) for t, p in zip(val_labels, val_preds))
        val_acc = correct / len(val_ds)
        # With the default per-split indexing, val indices can fall outside
        # the train vocabulary; name them so the F1 is still computable.
        val_true_names = [id2label.get(i, f"<val index {i}>") for i in val_labels]
        val_f1 = window_macro_f1(
            val_true_names,
            [id2label[i] for i in val_preds],
            classes=sorted(set(id2label.values()) | set(val_true_names)),
        )
        score = val_f1 if selection_metric == "macro_f1" else val_acc
        marker = " *" if score > best_val_acc else ""
        print(
            f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}"
            f"  val_macro_f1={val_f1:.4f}{marker}"
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(epoch_metrics)
        if epoch_callback is not None:
            epoch_callback(epoch_metrics)

        if score >= best_val_acc:
            best_val_acc = score
            best_epoch = epoch + 1
            if save_checkpoint:
                model.save_pretrained(output_dir / "best")
                processor.save_pretrained(output_dir / "best")

    print(
        f"\nDone. Best val {selection_metric}={best_val_acc:.4f}"
        f" (epoch {best_epoch}) — saved to {output_dir / 'best'}"
    )
    return {
        "selection_metric": selection_metric,
        "best_score": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


def _class_weights(dataset: BehaviorClipDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency loss weights, ``n / (k * n_c)``, over a dataset's clips."""
    counts = torch.bincount(
        torch.tensor([label for _, label in dataset.clips], dtype=torch.long),
        minlength=num_classes,
    ).float()
    return counts.sum() / (num_classes * counts.clamp(min=1))


def predict_clip_indices(
    model: VideoMAEForVideoClassification,
    loader: DataLoader,
    dev: torch.device,
    desc: str = "predict",
) -> tuple[list[int], list[int]]:
    """Run a classifier over a clip loader.

    Args:
        model: The classifier, already on ``dev``.
        loader: Loader over a :class:`BehaviorClipDataset`.
        dev: Device to run on.
        desc: Progress-bar label.

    Returns:
        True and predicted label indices, in loader order.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            outputs = model(pixel_values=batch["pixel_values"].to(dev))
            all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    return all_labels, all_preds


def update_train_video_model(
    base_path: Path,
    video_crop_path: Path,
    model_checkpoint: str = "MCG-NJU/videomae-base",
    num_frames: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    freeze_encoder: bool = True,
    run_name: str = "videomae_behavior",
) -> None:
    """Fine-tune VideoMAE into a fresh run directory and save the resolved config.

    Builds a :class:`ClassificationConfig` from the given arguments, resolves
    a new timestamped run directory under ``base_path``, trains into it, then
    writes the resolved config — including the path to the best checkpoint —
    to ``run_config.toml`` in the run directory, so each run records exactly
    what produced it.

    Args:
        base_path: Root directory under which a new timestamped run
            directory is created.
        video_crop_path: Root directory containing ``train/`` and ``val/``
            clip splits, as produced by
            :func:`bird_cv.classification.extract_behavior_clips.extract_behavior_clips`.
        model_checkpoint: HuggingFace model ID or local path to load
            VideoMAE weights from. Defaults to ``"MCG-NJU/videomae-base"``.
        num_frames: Frames per clip. Defaults to 16.
        epochs: Number of training epochs. Defaults to 10.
        batch_size: Dataloader batch size. Defaults to 8.
        lr: Learning rate. Defaults to 1e-4.
        device: Device string passed to ``torch.device``. Defaults to
            ``"cuda"``.
        freeze_encoder: If ``True``, freeze the VideoMAE encoder and only
            train the classification head. Defaults to ``True``.
        run_name: Name of the run — a subdirectory with this name is
            created under the run directory. Defaults to
            ``"videomae_behavior"``.
    """
    paths = Paths(
        base_path=str(base_path),
        video_crop_path=str(video_crop_path),
        model_checkpoint=model_checkpoint,
    )
    training = Training(
        num_frames=num_frames,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        freeze_encoder=freeze_encoder,
        run_name=run_name,
    )
    cfg = ClassificationConfig(paths=paths, training=training)
    run_dir = resolve_run_dir(Path(base_path), None)

    cfg.paths.output_root = str(run_dir)
    cfg.paths.best_checkpoint = str(run_dir / cfg.training.run_name / "best")

    # Save the config in run_dir
    with open(run_dir / "run_config.toml", "wb") as f:
        tomli_w.dump(msgspec.to_builtins(cfg), f)

    train_video_model(
        clips_root=Path(cfg.paths.video_crop_path),
        output_root=run_dir,
        output_name=cfg.training.run_name,
        model_checkpoint=cfg.paths.model_checkpoint,
        num_frames=cfg.training.num_frames,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        device=cfg.training.device,
        freeze_encoder=cfg.training.freeze_encoder,
    )
