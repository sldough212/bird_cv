"""Train a VideoMAE-based behavior classification model."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from transformers import (
    AutoImageProcessor,
    VideoMAEForVideoClassification,
)


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
    """

    def __init__(
        self,
        clips_root: Path,
        processor: AutoImageProcessor,
        num_frames: int = 16,
    ) -> None:
        self.processor = processor
        self.num_frames = num_frames

        self.label_dirs = sorted([d for d in clips_root.iterdir() if d.is_dir()])
        self.label_to_idx = {d.name: i for i, d in enumerate(self.label_dirs)}

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
        frames = [read_image(str(p)) for p in frame_paths]  # (C, H, W) uint8

        # processor expects list of PIL or numpy HWC — convert from CHW uint8
        frames_np = [f.permute(1, 2, 0).numpy() for f in frames]
        inputs = self.processor(frames_np, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }


def train_video_model(
    clips_root: Path,
    output_root: Path,
    output_name: str,
    model_checkpoint: str = "MCG-NJU/videomae-base",
    num_frames: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    freeze_encoder: bool = True,
) -> None:
    """Fine-tune a VideoMAE model for behavior classification.

    Loads the train and val splits from ``clips_root``, fine-tunes a
    VideoMAE classification head (and optionally the full encoder), and
    saves the best checkpoint by validation accuracy.

    Args:
        clips_root: Root directory containing ``train/`` and ``val/``
            split subdirectories produced by :func:`extract_behavior_clips`.
        output_root: Directory where run outputs will be saved.
        output_name: Name of the run — a subdirectory with this name is
            created under ``output_root``.
        model_checkpoint: HuggingFace model ID or local path to load
            VideoMAE weights from. Defaults to ``"MCG-NJU/videomae-base"``.
        num_frames: Frames per clip — must match what was used in
            :func:`extract_behavior_clips`. Defaults to 16.
        epochs: Number of training epochs. Defaults to 10.
        batch_size: Dataloader batch size. Defaults to 8.
        lr: Learning rate. Defaults to 1e-4.
        device: Device string passed to ``torch.device``. Defaults to
            ``"cuda"``.
        freeze_encoder: If ``True``, freeze the VideoMAE encoder and only
            train the classification head. Defaults to ``True``.
    """
    output_dir = output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(
        model_checkpoint, num_frames=num_frames
    )

    train_ds = BehaviorClipDataset(clips_root / "train", processor, num_frames)
    val_ds = BehaviorClipDataset(clips_root / "val", processor, num_frames)

    label_to_idx = train_ds.label_to_idx
    id2label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

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

    model = model.to(dev)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(dev)
            labels = batch["labels"].to(dev)

            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(dev)
                labels = batch["labels"].to(dev)
                outputs = model(pixel_values=pixel_values)
                correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()

        val_acc = correct / len(val_ds)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"loss={train_loss / len(train_loader):.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            print(f"  Saved best model (val_acc={best_val_acc:.4f})")
