# Run cropping
import argparse
from pathlib import Path

from bird_cv.client.post_behavior_predictions import post_behavior_predictions
from bird_cv.pipelines.config import BirdCVConfig
from bird_cv.pipelines.config import load_config


def run_classify(cfg: BirdCVConfig, hostname: str) -> None:
    post_behavior_predictions(
        host=hostname,
        port=8080,
        api_key=cfg.api_key,
        project_name=cfg.classify_project_name,
        model_path=cfg.classify.path_to_mae,
        predict_labeled=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the behavior classification pipeline."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file (e.g. configs/config.toml)",
    )
    parser.add_argument("hostname", type=str, help="Node IP hostname")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    run_classify(cfg, hostname=args.hostname)
