"""A tiny random V-JEPA 2 trains with a frozen encoder and runs windowed inference."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    VideoMAEConfig,
    VJEPA2Config,
    VJEPA2ForVideoClassification,
    VJEPA2VideoProcessor,
)

from bird_cv.classification.predict_video_model import VideoClassifier
from bird_cv.classification.train_video_model import train_video_model

SIZE, FRAMES = 32, 4


def _tiny_vjepa2(path: Path) -> None:
    config = VJEPA2Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        mlp_ratio=1,
        crop_size=16,  # trained at SIZE: the override must take effect
        frames_per_clip=FRAMES,
        pred_hidden_size=32,
        pred_num_hidden_layers=1,
        pred_num_attention_heads=2,
        num_pooler_layers=1,
    )
    VJEPA2ForVideoClassification(config).save_pretrained(path)
    VJEPA2VideoProcessor(crop_size=16).save_pretrained(path)


def _clips(root: Path) -> None:
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for label in ("resting", "flying"):
            for c in range(2):
                clip = root / split / label / f"c{c}_clip000"
                clip.mkdir(parents=True)
                for i in range(FRAMES):
                    frame = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
                    Image.fromarray(frame).save(clip / f"{i:05d}.jpg")


def test_frozen_vjepa2_trains_head_and_predicts(tmp_path: Path) -> None:
    pretrained = tmp_path / "pretrained"
    _tiny_vjepa2(pretrained)
    _clips(tmp_path / "clips")

    train_video_model(
        tmp_path / "clips",
        tmp_path / "run",
        pretrained,
        output_name="model",
        num_frames=FRAMES,
        epochs=1,
        batch_size=2,
        lr=1e-2,
        device="cpu",
        backbone="vjepa2",
        input_size=SIZE,
    )

    best = tmp_path / "run" / "model" / "best"
    before = VJEPA2ForVideoClassification.from_pretrained(pretrained).state_dict()
    after = VJEPA2ForVideoClassification.from_pretrained(best).state_dict()
    changed = {k.split(".")[0] for k in after if not torch.equal(before[k], after[k])}
    assert changed == {"pooler", "classifier"}

    classifier = VideoClassifier(
        str(best), num_frames=FRAMES, stride=2, input_size=SIZE, backbone="vjepa2"
    )
    frames = [np.zeros((SIZE, SIZE, 3), dtype=np.uint8)] * 10
    preds = classifier.predict_frames(frames)
    assert len(preds) == 10
    assert {p.label for p in preds} <= {"resting", "flying"}

    with pytest.raises(ValueError, match="48 px|input_size"):
        VideoClassifier(str(best), num_frames=FRAMES, input_size=48, backbone="vjepa2")

    # A leftover VideoMAE checkpoint is refused, not loaded with random weights
    VideoMAEConfig().save_pretrained(tmp_path / "videomae")
    with pytest.raises(ValueError, match="videomae checkpoint"):
        VideoClassifier(str(tmp_path / "videomae"), backbone="vjepa2")
