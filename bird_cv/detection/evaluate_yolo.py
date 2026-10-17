from pathlib import Path

import polars as pl
from ultralytics import YOLO

from bird_cv.evaluation.metrics import BOX_COLUMNS, detection_presence_metrics

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
    for _ in model.track(
        source=str(source_path),
        tracker=str(tracker_config),
        save_txt=True,
        save_conf=True,
        project=str(output_path.parent),
        name=output_path.name,
        exist_ok=True,
        stream=True,
        max_det=1,
    ):
        pass


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


def _yolo_label_box(label_file: Path) -> tuple[float, ...] | None:
    """First box of a YOLO label file as normalized (left, top, width, height)."""
    if not label_file.exists():
        return None
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cx, cy, w, h = map(float, parts[1:5])
            return (cx - w / 2, cy - h / 2, w, h)
    return None


def _boxes_frame(rows: list[tuple[str, tuple[float, ...] | None]]) -> pl.DataFrame:
    data: dict[str, list] = {"image": [r[0] for r in rows]}
    for i, col in enumerate(BOX_COLUMNS):
        data[col] = [r[1][i] if r[1] else None for r in rows]
    return pl.DataFrame(
        data, schema={"image": pl.String} | {c: pl.Float64 for c in BOX_COLUMNS}
    )


def yolo_split_boxes(
    dataset_root: Path,
    model_path: str | Path,
    split: str = "val",
    conf: float = 0.25,
    device: int | str | None = None,
    batch: int = 32,
    image_suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ground-truth and predicted boxes for every image of a YOLO dataset split.

    Images are read recursively from ``dataset_root/images/{split}`` and
    labels from the mirrored path under ``dataset_root/labels/{split}``; an
    empty or missing label file means no bird. Prediction keeps at most one
    box per image (a cage holds at most one bird).

    Args:
        dataset_root: YOLO dataset root containing ``images/`` and ``labels/``.
        model_path: YOLO weights to predict with.
        split: Split subdirectory. Defaults to ``"val"``.
        conf: Detection confidence threshold. Defaults to 0.25.
        device: Device passed to ``model.predict``; ``None`` lets YOLO choose.
        batch: Images per predict call.
        image_suffixes: Lower-case file suffixes counted as images.

    Returns:
        ``(truth, pred)``: one row per image keyed by ``image`` (path
        relative to the split directory) with
        :data:`bird_cv.evaluation.metrics.BOX_COLUMNS`, null when no box.
    """
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    images = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in image_suffixes
    )
    model = YOLO(model_path)

    truth_rows: list[tuple[str, tuple[float, ...] | None]] = []
    pred_rows: list[tuple[str, tuple[float, ...] | None]] = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            conf=conf,
            max_det=1,
            device=device,
            verbose=False,
        )
        for image, result in zip(chunk, results):
            rel = image.relative_to(images_dir)
            label_file = (labels_dir / rel).with_suffix(".txt")
            truth_rows.append((str(rel), _yolo_label_box(label_file)))
            box: tuple[float, ...] | None = None
            if result.boxes is not None and len(result.boxes):
                cx, cy, w, h = result.boxes.xywhn[0].tolist()
                box = (cx - w / 2, cy - h / 2, w, h)
            pred_rows.append((str(rel), box))

    return _boxes_frame(truth_rows), _boxes_frame(pred_rows)


def yolo_presence_metrics(
    dataset_root: Path,
    model_path: str | Path,
    split: str = "val",
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    device: int | str | None = None,
) -> dict[str, float | int]:
    """Bird presence precision/recall/F1 and mean IoU of a YOLO model on a split.

    Used to select YOLO checkpoints on validation presence F1 (ADR 0005).

    Args:
        dataset_root: YOLO dataset root containing ``images/`` and ``labels/``.
        model_path: YOLO weights to evaluate.
        split: Split subdirectory. Defaults to ``"val"``.
        conf: Detection confidence threshold. Defaults to 0.25.
        iou_threshold: Minimum IoU for a hit. Defaults to 0.5.
        device: Device passed to ``model.predict``.

    Returns:
        See :func:`bird_cv.evaluation.metrics.detection_presence_metrics`.
    """
    truth, pred = yolo_split_boxes(dataset_root, model_path, split, conf, device)
    return detection_presence_metrics(
        truth, pred, iou_threshold=iou_threshold, on=("image",)
    )
