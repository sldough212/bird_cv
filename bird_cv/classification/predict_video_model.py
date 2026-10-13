"""Windowed VideoMAE inference: load a model and predict behavior segments for a video."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

from bird_cv.classification.types import FramePrediction


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


def _predict_frame_votes(
    video_path: Path,
    model: VideoMAEForVideoClassification,
    processor: AutoImageProcessor,
    dev: torch.device,
    num_frames: int = 16,
    stride: int = 8,
    batch_size: int = 4,
) -> list[list[int]]:
    """Run windowed VideoMAE inference and collect per-frame class-id votes.

    Slides a ``num_frames``-frame window across the video and classifies
    each window. Shared by :func:`predict_video_segments` (which collapses
    the result to majority-voted segments) and :func:`predict_video_frames`
    (which keeps one row per frame).

    Args:
        video_path: Path to the video file.
        model: Loaded VideoMAE classification model, already on ``dev``.
        processor: HuggingFace image processor for the VideoMAE model.
        dev: Device the model is loaded on.
        num_frames: Frames per window. Must match the model's input size.
            Defaults to 16.
        stride: Step size between consecutive windows in frames. Defaults to 8.
        batch_size: Dataloader batch size for windowed inference. Defaults to 4.

    Returns:
        List with one entry per video frame, each a list of the class ids
        voted for that frame by every window covering it (empty if no
        window covers that frame).
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

    return frame_votes


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
    frame_votes = _predict_frame_votes(
        video_path, model, processor, dev, num_frames, stride, batch_size
    )
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


def predict_video_frames(
    video_path: Path,
    model: VideoMAEForVideoClassification,
    processor: AutoImageProcessor,
    dev: torch.device,
    id_to_label: dict,
    num_frames: int = 16,
    stride: int = 8,
    batch_size: int = 4,
) -> list[FramePrediction]:
    """Run windowed VideoMAE inference and return one prediction per frame.

    Unlike :func:`predict_video_segments`, which collapses the majority
    vote into contiguous runs, this keeps one row per video frame -- the
    shape needed for frame-aligned output (e.g. merging with frames that
    have no detection at all) and for frame-level evaluation against
    ground truth.

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
        One :class:`FramePrediction` per video frame, in frame order.
        ``label`` is ``None`` for frames no window covers (the first/last
        few frames when the video is shorter than one window past the
        last stride step); ``confidence`` is the majority label's vote
        share among windows covering that frame (0.0 when ``label`` is
        ``None``).
    """
    frame_votes = _predict_frame_votes(
        video_path, model, processor, dev, num_frames, stride, batch_size
    )

    predictions: list[FramePrediction] = []
    for frame, votes in enumerate(frame_votes):
        if not votes:
            predictions.append(FramePrediction(frame=frame, label=None, confidence=0.0))
            continue
        pred_id, count = Counter(votes).most_common(1)[0]
        predictions.append(
            FramePrediction(
                frame=frame,
                label=id_to_label[pred_id],
                confidence=count / len(votes),
            )
        )

    return predictions


@dataclass
class VideoMAEClassifier:
    """:class:`bird_cv.classification.types.BehaviorClassifier` backed by VideoMAE.

    Wraps model loading plus :func:`predict_video_frames` so classifiers
    are swappable: anything that implements
    ``predict(video_path) -> list[FramePrediction]`` can stand in for this
    in the single-video pipeline and its evaluation harness.

    Attributes:
        model_path: Path to the VideoMAE model weights.
        num_frames: Frames per window. Must match the model's input size.
        stride: Step size between consecutive windows in frames.
        batch_size: Dataloader batch size for windowed inference.
    """

    model_path: str
    num_frames: int = 16
    stride: int = 8
    batch_size: int = 4

    def __post_init__(self) -> None:
        self.model, self.processor, self.device, self.id_to_label = load_videomae_model(
            self.model_path, num_frames=self.num_frames
        )

    def predict(self, video_path: Path) -> list[FramePrediction]:
        return predict_video_frames(
            video_path=video_path,
            model=self.model,
            processor=self.processor,
            dev=self.device,
            id_to_label=self.id_to_label,
            num_frames=self.num_frames,
            stride=self.stride,
            batch_size=self.batch_size,
        )
