from pathlib import Path

from ultralytics import YOLO

TRACKER_CONFIG = Path(__file__).parent / "tracker.yaml"


def evaluate_yolo(
    source_path: Path,
    output_path: Path,
    model_path: str | Path = "yolo11n.pt",
    tracker_config: str | Path = TRACKER_CONFIG,
) -> None:
    """Run YOLO tracking on a single image directory and save per-frame txt results.

    Args:
        source_path: Directory of images to run tracking on.
        output_path: Directory where tracking results will be saved. Per-frame
            txt files are written to ``output_path/labels/``.
        model_path: Path to the YOLO model weights file. Defaults to "yolo11n.pt".
        tracker_config: Path to the BoT-SORT/ByteTrack tracker config yaml.
            Defaults to the bundled ``tracker.yaml``.
    """
    model = YOLO(model_path)
    model.track(
        source=str(source_path),
        tracker=str(tracker_config),
        save_txt=True,
        save_conf=True,
        project=str(output_path.parent),
        name=output_path.name,
        exist_ok=True,
        stream=True,
        max_det=1,
    )


def run_tracking_on_test(
    yolo_crop_path: Path,
    output_root: Path,
    model_path: str | Path = "yolo11n.pt",
    tracker_config: str | Path = TRACKER_CONFIG,
) -> None:
    """Run YOLO tracking on every camera/video/cage in the test split.

    Iterates over ``yolo_crop_path/images/test/{camera_id}/{video_id}/{cage_id}``
    and runs tracking on each cage directory, saving results to
    ``output_root/{camera_id}/{video_id}/{cage_id}/``.

    Args:
        yolo_crop_path: Root of the cropped YOLO dataset (contains
            ``images/`` and ``labels/`` subdirectories).
        output_root: Root directory where per-cage tracking results are saved.
        model_path: Path to the YOLO model weights file.
        tracker_config: Path to the tracker config yaml.
    """
    test_images_path = yolo_crop_path / "images" / "test"

    for camera_dir in sorted(test_images_path.iterdir()):
        camera_id = camera_dir.name
        for video_dir in sorted(camera_dir.iterdir()):
            video_id = video_dir.name
            for cage_dir in sorted(video_dir.iterdir()):
                cage_id = cage_dir.name
                print(f"Tracking {camera_id}/{video_id}/{cage_id}")
                evaluate_yolo(
                    source_path=cage_dir,
                    output_path=output_root / camera_id / video_id / cage_id,
                    model_path=model_path,
                    tracker_config=tracker_config,
                )
