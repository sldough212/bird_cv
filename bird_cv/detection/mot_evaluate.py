"""MOT-format conversion and metric computation for per-cage YOLO tracking evaluation."""

import re
from pathlib import Path

import motmetrics as mm
import numpy as np
import polars as pl
from PIL import Image


def _frame_num_from_name(filename: str) -> int:
    """Extract the video frame number from a cropped image/label filename.

    Expects names of the form ``{camera}.{video}_frame_{N}_cage_{C}.{ext}``.
    """
    match = re.search(r"_frame_(\d+)_cage_", filename)
    if match is None:
        raise ValueError(f"Cannot extract frame number from filename: {filename}")
    return int(match.group(1))


def yolo_track_to_mot(
    labels_path: Path,
    images_path: Path,
) -> np.ndarray:
    """Convert YOLO per-frame tracking txt files to an in-memory MOT array.

    YOLO tracking saves one txt file per frame under ``labels_path/``. Each
    line has the format::

        class_id  x_center  y_center  width  height  conf  track_id

    Returns an ``(N, 10)`` float array in MOTChallenge format::

        frame_id  track_id  left  top  width  height  conf  -1  -1  -1

    Frames with no detections contribute no rows. If a label file is missing
    for a frame that has a corresponding image, that frame is treated as having
    no detections.

    Args:
        labels_path: Directory containing per-frame ``.txt`` tracking output.
        images_path: Directory containing the corresponding frame images (used
            to resolve image dimensions for coordinate de-normalisation).
    """
    rows: list[list[float]] = []

    # Resolve image size once — all crops for a cage share the same dimensions
    sample_image = next(images_path.glob("*.jpg"), None) or next(
        images_path.glob("*.png")
    )
    with Image.open(sample_image) as img:
        img_width, img_height = img.size

    for label_file in sorted(labels_path.glob("*.txt")):
        frame_num = _frame_num_from_name(label_file.stem)
        lines = label_file.read_text().strip().splitlines()
        for line in lines:
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 7:
                continue
            _class_id, x_ctr, y_ctr, w, h, conf, track_id = parts[:7]
            left = (x_ctr - w / 2) * img_width
            top = (y_ctr - h / 2) * img_height
            bb_w = w * img_width
            bb_h = h * img_height
            rows.append([frame_num, track_id, left, top, bb_w, bb_h, conf, -1, -1, -1])

    if not rows:
        return np.empty((0, 10), dtype=float)
    return np.array(rows, dtype=float)


def yolo_gt_to_mot(
    labels_path: Path,
    images_path: Path,
) -> np.ndarray:
    """Convert YOLO ground-truth label files to an in-memory MOT array.

    GT label files have one detection per line in the format::

        class_id  x_center  y_center  width  height

    Since each cage contains at most one bird, track_id is always ``1``.
    Frames with empty label files (no bird present) contribute no rows.

    Returns an ``(N, 10)`` float array in MOTChallenge format::

        frame_id  track_id  left  top  width  height  conf  -1  -1  -1

    Args:
        labels_path: Directory containing per-frame YOLO ``.txt`` GT labels.
        images_path: Directory containing the corresponding frame images.
    """
    rows: list[list[float]] = []

    sample_image = next(images_path.glob("*.jpg"), None) or next(
        images_path.glob("*.png")
    )
    with Image.open(sample_image) as img:
        img_width, img_height = img.size

    for label_file in sorted(labels_path.glob("*.txt")):
        frame_num = _frame_num_from_name(label_file.stem)
        lines = label_file.read_text().strip().splitlines()
        for line in lines:
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 5:
                continue
            _class_id, x_ctr, y_ctr, w, h = parts[:5]
            left = (x_ctr - w / 2) * img_width
            top = (y_ctr - h / 2) * img_height
            bb_w = w * img_width
            bb_h = h * img_height
            rows.append([frame_num, 1, left, top, bb_w, bb_h, 1.0, -1, -1, -1])

    if not rows:
        return np.empty((0, 10), dtype=float)
    return np.array(rows, dtype=float)


def compute_mot_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
) -> dict[str, float]:
    """Compute MOT metrics for a single cage given GT and prediction MOT arrays.

    Args:
        gt: Ground-truth MOT array as returned by :func:`yolo_gt_to_mot`.
        pred: Prediction MOT array as returned by :func:`yolo_track_to_mot`.

    Returns:
        Dictionary of metric name → value.
    """
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = np.union1d(
        gt[:, 0].astype(int) if len(gt) else np.array([], dtype=int),
        pred[:, 0].astype(int) if len(pred) else np.array([], dtype=int),
    )

    for frame in all_frames:
        gt_dets = gt[gt[:, 0] == frame][:, 1:6] if len(gt) else np.empty((0, 5))
        pred_dets = pred[pred[:, 0] == frame][:, 1:6] if len(pred) else np.empty((0, 5))

        gt_ids = gt_dets[:, 0].astype(int).tolist()
        pred_ids = pred_dets[:, 0].astype(int).tolist()

        dist = mm.distances.iou_matrix(gt_dets[:, 1:], pred_dets[:, 1:], max_iou=0.5)
        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "recall",
            "precision",
            "num_objects",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mota",
            "motp",
        ],
        name="acc",
    )

    return summary.loc["acc"].to_dict()


def evaluate_tracking(
    yolo_crop_path: Path,
    tracking_output_root: Path,
    output_path: Path,
) -> pl.DataFrame:
    """Compute MOT metrics for every camera/video/cage in the test split.

    For each cage, converts YOLO tracking output and ground-truth labels to
    MOT format and computes tracking metrics. Results are saved as a parquet
    file and returned as a Polars DataFrame with one row per cage.

    Args:
        yolo_crop_path: Root of the cropped YOLO dataset. Ground-truth labels
            are read from ``yolo_crop_path/labels/test/{camera_id}/{video_id}/{cage_id}/``.
            Corresponding images are read from ``yolo_crop_path/images/test/...``.
        tracking_output_root: Root directory of YOLO tracking output as written
            by :func:`run_tracking_on_test`. Per-cage tracking txt files are
            expected at ``tracking_output_root/{camera_id}/{video_id}/{cage_id}/labels/``.
        output_path: Path where the evaluation results will be saved as a parquet file.

    Returns:
        Polars DataFrame with columns ``camera_id``, ``video_id``, ``cage_id``,
        and one column per MOT metric.
    """
    records = []
    gt_labels_root = yolo_crop_path / "labels" / "test"
    gt_images_root = yolo_crop_path / "images" / "test"

    for camera_dir in sorted(gt_labels_root.iterdir()):
        camera_id = camera_dir.name
        for video_dir in sorted(camera_dir.iterdir()):
            video_id = video_dir.name
            for cage_dir in sorted(video_dir.iterdir()):
                cage_id = cage_dir.name
                print(f"Evaluating {camera_id}/{video_id}/{cage_id}")

                pred_labels_path = (
                    tracking_output_root / camera_id / video_id / cage_id / "labels"
                )
                images_path = gt_images_root / camera_id / video_id / cage_id

                if not pred_labels_path.exists():
                    print(f"  Skipping: no tracking output found at {pred_labels_path}")
                    continue

                gt = yolo_gt_to_mot(
                    labels_path=cage_dir,
                    images_path=images_path,
                )
                pred = yolo_track_to_mot(
                    labels_path=pred_labels_path,
                    images_path=images_path,
                )

                metrics = compute_mot_metrics(gt=gt, pred=pred)
                records.append(
                    {"camera_id": camera_id, "video_id": video_id, "cage_id": cage_id}
                    | metrics
                )

    results = pl.DataFrame(records)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results.write_parquet(output_path)
    return results
