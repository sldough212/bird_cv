from bird_cv.client.evaluate_label_studio_tracking import evaluate_label_studio_tracking
from bird_cv.pipelines.config import BirdCVConfig, load_config

import argparse
from pathlib import Path


def get_tracking_metrics(cfg: BirdCVConfig, hostname: str) -> None:
    evaluate_label_studio_tracking(
        host=hostname,
        port=8080,
        api_key=cfg.api_key,
        project_name=cfg.detect_project_name,
        output_path=Path("test_metrics.parquet"),
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
    get_tracking_metrics(cfg, hostname=args.hostname)
