from types import SimpleNamespace

import numpy as np
import torch

from bird_cv.classification.predict_video_model import predict_frame_list


def processor(window: list, return_tensors: str) -> dict:
    return {"pixel_values": torch.zeros(1, len(window))}


def model(pixel_values: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(logits=torch.zeros(len(pixel_values), 2))


def test_every_frame_of_a_900_frame_input_is_labeled() -> None:
    # 900 frames, 16-frame windows every 8 frames: the stride alone stops at
    # frame 895, so cover_tail must add a window ending on frame 899.
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 900
    preds = predict_frame_list(
        frames, model, processor, torch.device("cpu"), {0: "resting"}
    )
    assert [p.frame for p in preds] == list(range(900))
    assert all(p.label == "resting" for p in preds)
