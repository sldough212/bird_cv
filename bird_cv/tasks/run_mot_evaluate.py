from pathlib import Path

from bird_cv.detection.mot_evaluate import evaluate_tracking

if __name__ == "__main__":
    evaluate_tracking(
        yolo_crop_path=Path("/gscratch/pdoughe1/20260331_194037/yolo_crop"),
        tracking_output_root=Path("/gscratch/pdoughe1/20260331_194037/tracking"),
        output_path=Path(
            "/gscratch/pdoughe1/20260331_194037/mot_evaluation/results.parquet"
        ),
    )
