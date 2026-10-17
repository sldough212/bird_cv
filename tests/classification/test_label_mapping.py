"""Val/test splits share train's label indices; YOLO tuning asks for a GPU."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from bird_cv.classification import train_video_model as tvm
from bird_cv.detection import train_yolo as ty


def _make_clips(split_root: Path, labels: list[str]) -> None:
    for label in labels:
        clip = split_root / label / "t1_c1_clip0"
        clip.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
                clip / f"{i:05d}.jpg"
            )


def test_val_missing_class_uses_train_indices(tmp_path, monkeypatch):
    _make_clips(tmp_path / "train", ["eat", "preen", "sleep"])
    _make_clips(tmp_path / "val", ["eat", "sleep", "unseen"])
    built: list[tvm.BehaviorClipDataset] = []

    class RecordingDataset(tvm.BehaviorClipDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    def stub_processor(frames, return_tensors):
        return {"pixel_values": torch.zeros(1, len(frames), 3, 16, 16)}

    class StopBeforeModel(Exception):
        pass

    monkeypatch.setattr(tvm, "BehaviorClipDataset", RecordingDataset)
    monkeypatch.setattr(
        tvm.AutoImageProcessor, "from_pretrained", lambda *a, **k: stub_processor
    )
    monkeypatch.setattr(
        tvm.VideoMAEForVideoClassification,
        "from_pretrained",
        MagicMock(side_effect=StopBeforeModel),
    )

    with pytest.raises(StopBeforeModel):
        tvm.train_video_model(tmp_path, tmp_path / "out", tmp_path, device="cpu")

    train_ds, val_ds = built
    assert val_ds.label_to_idx == train_ds.label_to_idx
    assert [label for _, label in val_ds.clips] == [0, 2]  # eat, sleep
    assert val_ds.n_skipped_clips == 1
    assert val_ds[1]["labels"].item() == 2


def test_train_yolo_tune_requests_one_gpu_per_trial(tmp_path, monkeypatch):
    yolo = MagicMock()
    monkeypatch.setattr(ty, "YOLO", yolo)

    ty.train_yolo(tmp_path, "run", tune=True)

    assert yolo.return_value.tune.call_args.kwargs["gpu_per_trial"] == 1
