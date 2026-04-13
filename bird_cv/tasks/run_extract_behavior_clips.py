from bird_cv.classification.extract_behavior_clips import extract_behavior_clips
from pathlib import Path


extract_behavior_clips(
    behavior_index_path=Path(
        "/gscratch/pdoughe1/20260412_221017/intermediate/behavior_index.parquet"
    ),
    cropped_frames_path=Path("/gscratch/pdoughe1/20260412_221017/yolo_crop"),
    output_path=Path("/gscratch/pdoughe1/20260412_221017/video_crop"),
    num_frames=16,
    stride=8,
)
