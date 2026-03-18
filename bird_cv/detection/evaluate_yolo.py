from ultralytics import YOLO
from pathlib import Path


def evaluate_yolo(
    source_path: Path,
    output_root: Path,
    output_name: str,
    tracker_config: str = "tracker.yaml",
    model_path: str | Path = "yolo11n.pt",
) -> None:
    """Runs YOLO object detection and tracking on a given source and saves results.

    This function loads a YOLO model and performs object tracking on the input
    data (e.g., video or image sequence). Tracking results, including bounding
    boxes and confidence scores, are saved as text files in the specified
    output directory.

    Args:
        source_path (Path): Path to the input source (e.g., video file, image
            directory, or stream).
        output_root (Path): Root directory where output results will be stored.
        output_name (str): Name of the output subdirectory within the project
            folder.
        tracker_config (str, optional): Path to the tracker configuration file.
            Defaults to "tracker.yaml".
        model_path (str | Path, optional): Path to the YOLO model weights file.
            Defaults to "yolo11n.pt".

    Returns:
        None: This function does not return a value. Results are written to disk.

    Notes:
        - Tracking is performed with persistence enabled across frames.
        - Bounding box coordinates and confidence scores are saved as text files.
        - Visualization images are not saved (`save=False`) and not displayed
          (`show=False`).
    """
    model = YOLO(model_path)

    model.track(
        source=source_path,
        tracker=tracker_config,
        persist=True,
        save=False,  # Saves the plots
        save_txt=True,
        save_conf=True,
        project=output_root,
        name=output_name,
        show=False,
    )


if __name__ == "__main__":
    evaluate_yolo(
        source_path=Path(
            "/gscratch/pdoughe1/label_test/20260307_122739/yolo_store/test/images"
        ),
        output_root=Path("/gscratch/pdoughe1/yolo_train"),
        output_name="tracker_test",
        model_path=Path("/gscratch/pdoughe1/yolo_train/bird_test/weights/best.pt"),
        tracker_config=Path("bird_cv/detection/tracker.yaml"),
    )
