from bird_cv.detection.train_yolo import update_train_yolo
from bird_cv.pipelines.config import BirdCVConfig, load_config

import argparse
from pathlib import Path


def run_retrain_detect(cfg: BirdCVConfig) -> None:
    update_train_yolo(
        base_path=cfg.detect.path_to_output,
        previous_model_config=cfg.detect.path_to_yolo_config,
        pretrained_checkpoint=cfg.detect.path_to_yolo,
        yolo_training_data=cfg.detect.path_to_training_data,
        epochs=cfg.detect.epochs,
        device=cfg.detect.device,
        tune=cfg.detect.tune,
        tune_iterations=cfg.detect.tune_iterations,
        run_name=cfg.detect.run_name,  # is this really necessary?
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the detection pipeline.")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file (e.g. configs/config.toml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    run_retrain_detect(cfg)
