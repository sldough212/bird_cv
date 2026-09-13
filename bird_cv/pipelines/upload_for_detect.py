import argparse
from pathlib import Path

from bird_cv.segmentation.segment import run_segment
from bird_cv.client.upload_tasks import upload_video_tasks
from bird_cv.preprocessing.get_split_guidance import simulate_split_guidance
from bird_cv.preprocessing.crop import crop_cages
from bird_cv.preprocessing.image_utils import run_images_to_video
from bird_cv.pipelines.config import BirdCVConfig, load_config


def upload_for_detect(cfg: BirdCVConfig, hostname: str):
    simulate_split_guidance(
        videos_path=cfg.path_to_raw_videos,
        output_path=cfg.path_to_guidance,
    )

    run_segment(
        segmentation_configs_path=cfg.path_to_segmentation_configs,
        sam_model_id=cfg.sam_model_id,
        split_guidance_path=cfg.path_to_guidance,
        segmentations_path=cfg.path_to_segmentation_output,
        videos_path=cfg.path_to_raw_videos,
    )

    crop_cages(
        split_guidance_path=cfg.path_to_guidance,
        video_segments_path=cfg.path_to_segmentation_output,
        clip_output_path=cfg.path_to_cropped_cage_frames,
        videos_path=cfg.path_to_raw_videos,
    )

    # Save videos as mp4s
    run_images_to_video(
        split_guidance_path=cfg.path_to_guidance,
        clip_output_path=cfg.path_to_cropped_cage_frames,
        video_output_path=cfg.path_to_cage_videos,
    )

    # Upload the videos
    upload_video_tasks(
        port=8080,
        host=hostname,
        api_key=cfg.api_key,
        project_name=cfg.detect_project_name,
        video_path=cfg.path_to_cage_videos,
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
    upload_for_detect(cfg, hostname=args.hostname)
