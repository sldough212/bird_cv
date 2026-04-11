from pathlib import Path

from bird_cv.preprocessing.crop_yolo_labels import run_crop_yolo
from bird_cv.classification.build_clip_index import build_clip_index


base_path = Path("/gscratch/pdoughe1/20260411_161837")

build_clip_index(
    frame_data_path=base_path / "intermediate/frame_data_full.ndjson",
    video_data_path=base_path / "intermediate/video_data_full.ndjson",
    split_guidance_path=base_path / "intermediate/split_guidance.parquet",
    output_path=base_path / "intermediate/clip_index.parquet",
)

run_crop_yolo(
    split_guidance_path=base_path / "intermediate/split_guidance.parquet",
    yolo_data_path=base_path / "yolo_store",
    yolo_output_path=base_path / "yolo_crop",
    video_segments_path=base_path / "segmentations",
    behavior_clips_path=base_path / "intermediate/clip_index.parquet",
    behavior_output_path=base_path / "intermediate/behavior_index.parquet",
)
