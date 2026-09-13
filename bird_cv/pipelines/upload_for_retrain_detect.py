from bird_cv.preprocessing.get_split_guidance import split_guidance_within_camera
from bird_cv.preprocessing.annotations_to_yolo import stream_annotations_to_yolo
from bird_cv.pipelines.config import BirdCVConfig, load_config
from bird_cv.client.get_label_studio_annotations import get_label_studio_annotations

import argparse
from pathlib import Path


def upload_for_retrain_detect(cfg: BirdCVConfig, hostname: str):
    get_label_studio_annotations(
        host=hostname,
        port=8080,
        api_key=cfg.api_key,
        project_name=cfg.detect_project_name,
        output_path=cfg.path_to_detect_annotations,
        interpolate_frames=True,
    )

    split_guidance_within_camera(
        path_to_guidance=cfg.path_to_guidance,
        split_ratio={"train": 0.75, "val": 0.25},
        output_path=cfg.path_to_guidance_within_camera,
    )

    stream_annotations_to_yolo(
        path_to_videos=cfg.path_to_cage_videos,
        path_to_annotations=cfg.path_to_detect_annotations,
        path_to_guidance=cfg.path_to_guidance_within_camera,
        path_to_output=cfg.detect.path_to_training_data,
        processes=1,
        crop_size=None,
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
    upload_for_retrain_detect(cfg, hostname=args.hostname)
