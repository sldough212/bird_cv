import argparse
from pathlib import Path

from bird_cv.pipelines.config import BirdCVConfig, load_config
from bird_cv.client.post_bbox_predictions import post_bbox_predictions


def run_detect(cfg: BirdCVConfig, hostname: str) -> None:
    post_bbox_predictions(
        port=8080,
        host=hostname,
        api_key=cfg.api_key,
        project_name=cfg.detect_project_name,
        model_path=cfg.detect.path_to_yolo,
        tracker_path=cfg.detect.path_to_tracker_config,
        predict_labeled=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the detection pipeline.")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file (e.g. configs/config.toml)",
    )
    parser.add_argument("hostname", type=str, help="Node IP hostname")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    run_detect(cfg, hostname=args.hostname)
