import argparse
from pathlib import Path

from bird_cv.client.get_label_studio_annotations import get_label_studio_annotations
from bird_cv.pipelines.config import BirdCVConfig, load_config


def upload_for_retrain_classify(cfg: BirdCVConfig, hostname: str) -> None:
    # Camera/video train-val split is shared with detect: this assumes
    # upload_for_retrain_detect.py has already been run at least once to
    # produce cfg.path_to_guidance_within_camera, so both pipelines train
    # and evaluate on the same per-video assignment.
    get_label_studio_annotations(
        host=hostname,
        port=8080,
        api_key=cfg.api_key,
        project_name=cfg.classify_project_name,
        output_path=cfg.path_to_classify_annotations,
        interpolate_frames=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Label Studio behavior annotations for retraining."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file (e.g. configs/config.toml)",
    )
    parser.add_argument("hostname", type=str, help="Node IP hostname")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    upload_for_retrain_classify(cfg, hostname=args.hostname)
