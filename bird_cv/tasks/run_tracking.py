from pathlib import Path

from bird_cv.detection.evaluate_yolo import run_tracking_on_test

if __name__ == "__main__":
    run_tracking_on_test(
        yolo_crop_path=Path("/gscratch/pdoughe1/20260331_194037/yolo_crop"),
        output_root=Path("/gscratch/pdoughe1/20260331_194037/tracking"),
        model_path=Path(
            "/gscratch/pdoughe1/20260331_194037/training/bird_test/weights/best.pt"
        ),
    )
