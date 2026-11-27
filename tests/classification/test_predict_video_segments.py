from types import SimpleNamespace

import numpy as np
import torch

from bird_cv.classification.predict_video_model import predict_video_segments


class FakeCapture:
    def __init__(self, n: int) -> None:
        self.frames = iter([np.zeros((2, 2, 3), dtype=np.uint8)] * n)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self) -> None:
        pass


def processor(window: list, return_tensors: str) -> dict:
    return {"pixel_values": torch.zeros(1, len(window))}


def model(pixel_values: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(logits=torch.zeros(len(pixel_values), 2))


def test_segments_cover_every_frame_of_a_900_frame_video(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "bird_cv.classification.predict_video_model.cv2.VideoCapture",
        lambda _: FakeCapture(900),
    )

    segments = predict_video_segments(
        tmp_path / "v.mp4", model, processor, torch.device("cpu"), {0: "resting"}
    )

    assert segments == [("resting", 0, 899)]
