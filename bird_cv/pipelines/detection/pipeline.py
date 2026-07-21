"""Config schema for YOLO detection training runs.

Populated and serialized by :func:`bird_cv.detection.train_yolo.update_train_yolo`,
which writes the resolved config alongside each run's outputs for reproducibility.
"""

import msgspec


class Paths(msgspec.Struct):
    base_path: str
    model_config: str
    pretrained_checkpoint: str
    output_root: str = ""
    new_checkpoint: str = ""


class Training(msgspec.Struct):
    epochs: int = 30
    device: int = 0
    tune: bool = False
    tune_iterations: int = 30
    run_name: str = "bird_yolo"


class DetectionConfig(msgspec.Struct):
    paths: Paths
    training: Training
