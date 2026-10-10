# Run cropping
import argparse
from pathlib import Path

from bird_cv.preprocessing.annotations_to_yolo import stream_annotations_to_yolo
from bird_cv.preprocessing.image_utils import run_images_to_video
from bird_cv.client.upload_tasks import upload_video_tasks
from bird_cv.pipelines.config import BirdCVConfig
from bird_cv.client.get_label_studio_annotations import get_label_studio_annotations
from bird_cv.pipelines.config import load_config


def upload_for_classify(cfg: BirdCVConfig, hostname: str) -> None:
    # Download annotations
    get_label_studio_annotations(
        host=hostname,
        port=8080,
        api_key=cfg.api_key,
        project_name=cfg.detect_project_name,
        output_path=cfg.path_to_detect_annotations,
        interpolate_frames=True,
    )

    stream_annotations_to_yolo(
        path_to_videos=cfg.path_to_cage_videos,
        path_to_annotations=cfg.path_to_detect_annotations,
        path_to_guidance=cfg.path_to_guidance,
        path_to_output=cfg.path_to_cropped_bbox_frames,
        processes=1,
        crop_size=224,
    )

    run_images_to_video(
        split_guidance_path=cfg.path_to_guidance,
        clip_output_path=cfg.path_to_cropped_bbox_frames / "images" / "train",
        video_output_path=cfg.path_to_bbox_videos,
    )

    # Upload crops
    upload_video_tasks(
        port=8080,
        host=hostname,
        api_key=cfg.api_key,
        project_name=cfg.classify_project_name,
        video_path=cfg.path_to_bbox_videos,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load clips into behavior classification pipeline."
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the TOML config file (e.g. configs/config.toml)",
    )
    parser.add_argument("hostname", type=str, help="Node IP hostname")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    upload_for_classify(cfg, hostname=args.hostname)
