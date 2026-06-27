from pathlib import Path

from bird_cv.pipelines.segment_videos.segment_videos import run_images_to_video

# Define paths
split_guidance_path = Path(
    "/Users/sdougherty/Documents/code/data/ml_ls_demo/split_guidance.parquet"
)
clip_output_path = Path("scratch") / "20260627" / "images" / "train"
video_output_path = Path("scratch") / "20260627" / "videos"

# Convert crops to videos
run_images_to_video(
    split_guidance_path=split_guidance_path,
    clip_output_path=clip_output_path,
    video_output_path=video_output_path,
)
