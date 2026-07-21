"""Config schema for VideoMAE behavior classification training runs.

Populated and serialized by
:func:`bird_cv.classification.train_video_model.update_train_video_model`,
which writes the resolved config alongside each run's outputs for reproducibility.
"""

import msgspec


class Paths(msgspec.Struct):
    base_path: str
    video_crop_path: str
    model_checkpoint: str
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
