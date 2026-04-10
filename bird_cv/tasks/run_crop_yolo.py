from pathlib import Path

from bird_cv.preprocessing.crop_yolo_labels import run_crop_yolo
from bird_cv.classification.build_clip_index import build_clip_index


build_clip_index(
    frame_data_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/frame_data_full.ndjson"
    ),
    video_data_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/video_data_full.ndjson"
    ),
    split_guidance_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/corrected_frame_guidance"
    ),
    output_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/clip_index.parquet"
    ),
)

run_crop_yolo(
    corrected_targets_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/corrected_frame_guidance"
    ),
    yolo_data_path=Path("/gscratch/pdoughe1/20260409_180455/yolo_store"),
    yolo_output_path=Path("/gscratch/pdoughe1/20260409_180455/yolo_crop"),
    video_segments_path=Path("/gscratch/pdoughe1/20260409_180455/segmentations"),
    behavior_clips_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/clip_index.parquet"
    ),
    behavior_output_path=Path(
        "/gscratch/pdoughe1/20260409_180455/intermediate/behavior_index.parquet"
    ),
)
