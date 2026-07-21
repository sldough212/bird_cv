"""Windowed VideoMAE inference: load a model and predict behavior segments for a video."""

from collections import Counter
from pathlib import Path

import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, VideoMAEForVideoClassification


class VideoWindowDataset(Dataset):
    """Sliding window dataset over a single video for inference.

    Reads all frames into memory on construction, then yields ``num_frames``-frame
    windows with the given stride. Each item includes the processed pixel values
    and the window's start frame index for mapping predictions back to timestamps.

    Args:
        video_path: Path to the video file.
        processor: HuggingFace image processor for the VideoMAE model.
        num_frames: Number of frames per window. Must match the model's input size.
        stride: Step size between consecutive windows in frames.
    """

    def __init__(
        self,
        video_path: Path,
        processor: AutoImageProcessor,
        num_frames: int = 16,
        stride: int = 8,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        self.frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        self.processor = processor
        self.num_frames = num_frames
        self.start_frames = list(range(0, len(self.frames) - num_frames + 1, stride))

    def __len__(self) -> int:
        return len(self.start_frames)

    def __getitem__(self, idx: int) -> dict:
        start = self.start_frames[idx]
        window = self.frames[start : start + self.num_frames]
        inputs = self.processor(window, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "start_frame": start,
        }


def load_videomae_model(
    model_path: str, num_frames: int = 16
) -> tuple[VideoMAEForVideoClassification, AutoImageProcessor, torch.device, dict]:
    """Load a fine-tuned VideoMAE model and processor for inference.

    Args:
        model_path: Path to the VideoMAE model weights.
        num_frames: Number of frames per window the processor expects. Defaults to 16.

    Returns:
        Tuple of ``(model, processor, device, id_to_label)``, with the model
        already moved to ``device`` and set to eval mode.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_path, num_frames=num_frames)
    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    model = model.to(dev)
    model.eval()
    id_to_label = model.config.id2label
    return model, processor, dev, id_to_label


def predict_video_segments(
    video_path: Path,
    model: VideoMAEForVideoClassification,
    processor: AutoImageProcessor,
    dev: torch.device,
    id_to_label: dict,
    num_frames: int = 16,
    stride: int = 8,
    batch_size: int = 4,
) -> list[tuple[str, int, int]]:
    """Run windowed VideoMAE inference over a video and encode it into behavior segments.

    Slides a ``num_frames``-frame window across the video, classifies each
    window, takes a majority vote across windows covering each frame, then
    run-length encodes the per-frame labels into contiguous segments.

    Args:
        video_path: Path to the video file.
        model: Loaded VideoMAE classification model, already on ``dev``.
        processor: HuggingFace image processor for the VideoMAE model.
        dev: Device the model is loaded on.
        id_to_label: Mapping from predicted class id to label name.
        num_frames: Frames per window. Must match the model's input size.
            Defaults to 16.
        stride: Step size between consecutive windows in frames. Defaults to 8.
        batch_size: Dataloader batch size for windowed inference. Defaults to 4.

    Returns:
        List of ``(label, start_frame, end_frame)`` tuples, one per
        contiguous run of frames sharing the majority-voted label, in frame
        order.
    """
    ds = VideoWindowDataset(video_path, processor, num_frames, stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Collect per-window predictions alongside their start frames
    window_preds: list[tuple[int, int]] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(dev)
            outputs = model(pixel_values=pixel_values)
            pred_ids = outputs.logits.argmax(dim=-1).cpu().tolist()
            start_frames = batch["start_frame"].tolist()
            window_preds.extend(zip(start_frames, pred_ids))

    # Majority vote across all windows that cover each frame
    total_frames = len(ds.frames)
    frame_votes: list[list[int]] = [[] for _ in range(total_frames)]
    for start, pred_id in window_preds:
        for f in range(start, min(start + num_frames, total_frames)):
            frame_votes[f].append(pred_id)

    frame_labels = [
        Counter(votes).most_common(1)[0][0] if votes else None for votes in frame_votes
    ]

    # Run-length encode into contiguous segments
    segments = []
    if frame_labels:
        seg_start = 0
        current = frame_labels[0]
        for i, label in enumerate(frame_labels[1:], 1):
            if label != current:
                if current is not None:
                    segments.append((id_to_label[current], seg_start, i - 1))
                current = label
                seg_start = i
        if current is not None:
            segments.append((id_to_label[current], seg_start, len(frame_labels) - 1))

    return segments
