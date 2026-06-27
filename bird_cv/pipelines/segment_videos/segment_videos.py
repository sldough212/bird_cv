from pathlib import Path
from bird_cv.segmentation.segment import run_segment
from bird_cv.segment_videos.segment_videos import (
    simulate_split_guidance,
    crop_cages,
    run_images_to_video,
)

base_output = Path("/gscratch/pdoughe1/label_studio_test")
base_output.mkdir(exist_ok=True, parents=True)
intermediate_output = base_output / "intermediate"
intermediate_output.mkdir(exist_ok=True, parents=True)

videos_path = Path("/gscratch/pdoughe1/videos/label_studio_integration")
segmentation_configs_path = Path("/gscratch/pdoughe1/segmentation_configs/configs")
model_checkpoint_path = Path("/home/pdoughe1/sam2/checkpoints")

simulate_split_guidance(
    videos_path, output_path=intermediate_output / "split_guidance.parquet"
)

run_segment(
    segmentation_configs_path=segmentation_configs_path,
    model_checkpoint_path=model_checkpoint_path,
    split_guidance_path=intermediate_output / "split_guidance.parquet",
    segmentations_path=intermediate_output / "segmentations",
    videos_path=videos_path,
)

crop_cages(
    split_guidance_path=intermediate_output / "split_guidance.parquet",
    video_segments_path=intermediate_output / "segmentations",
    clip_output_path=base_output / "cropped_images",
    videos_path=videos_path,
)

# Save videos as mp4s
run_images_to_video(
    split_guidance_path=intermediate_output / "split_guidance.parquet",
    clip_output_path=base_output / "cropped_images",
    video_output_path=base_output / "cropped_videos",
)
