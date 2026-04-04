from pathlib import Path

from bird_cv.preprocessing.crop_yolo_labels import run_crop_yolo


seg_target_map = run_crop_yolo(
    corrected_targets_path=Path(
        "/gscratch/pdoughe1/20260331_194037/intermediate/corrected_frame_guidance"
    ),
    yolo_data_path=Path("/gscratch/pdoughe1/20260331_194037/yolo_store"),
    yolo_output_path=Path("/gscratch/pdoughe1/20260331_194037/yolo_crop"),
    video_segments_path=Path("/gscratch/pdoughe1/20260331_194037/segmentations"),
)
