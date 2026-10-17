"""Parquet schemas exchanged between pipeline stages.

Every stage that writes one of these tables builds it with the matching
schema (e.g. ``pl.DataFrame(rows, schema=PREDICTION_SCHEMA)``) so readers
never have to guess column names or types. All frame columns hold
canonical frames (see :mod:`bird_cv.core.frames`).
"""

import polars as pl

KEY_COLUMNS = ("camera_id", "video_id", "cage_id", "frame")
"""Columns that identify one cage on one canonical frame.

Formats (always build them with :mod:`bird_cv.core.ids`):

- ``camera_id``: decoded camera folder name, e.g. ``"H7,I22"`` (never ``%2C``).
- ``video_id``: raw video name without extension, e.g. ``"IMG_0016"``.
- ``cage_id``: cage index as a decimal string, e.g. ``"0"``.
- ``frame``: canonical frame.
"""

GROUND_TRUTH_SCHEMA = {
    "camera_id": pl.String,
    "video_id": pl.String,
    "cage_id": pl.String,
    "frame": pl.Int64,
    # A class from bird_cv.core.labels.CLASSES, or null for a labeling hole.
    "label": pl.String,
    # Annotated box normalized 0-1 to the cage video, top-left + size; null
    # when there is no box (label no_bird). Label Studio exports percentages,
    # so divide its x/y/width/height by 100.
    "bbox_left": pl.Float64,
    "bbox_top": pl.Float64,
    "bbox_width": pl.Float64,
    "bbox_height": pl.Float64,
}

PREDICTION_SCHEMA = {
    "camera_id": pl.String,
    "video_id": pl.String,
    "cage_id": pl.String,
    "frame": pl.Int64,
    # Never null: frames with no usable detection, or not covered by any
    # classifier window, are no_bird.
    "label": pl.String,
    # Classifier confidence, 0-1; null when label is no_bird.
    "confidence": pl.Float64,
    # Box normalized 0-1 to the cage video, top-left + size; null when no_bird.
    # Gap-filled frames carry the source detection's box and det_conf.
    "bbox_left": pl.Float64,
    "bbox_top": pl.Float64,
    "bbox_width": pl.Float64,
    "bbox_height": pl.Float64,
    "det_conf": pl.Float64,
}

SPLIT_SCHEMA = {
    "camera_id": pl.String,
    # "train", "val" or "test".
    "split": pl.String,
    # Split version in which this camera was first assigned.
    "assigned_in_version": pl.Int64,
}

CV_FOLD_SCHEMA = {
    "camera_id": pl.String,
    # 0-indexed fold in which this camera is the test set.
    "fold": pl.Int64,
}

SPLIT_SUMMARY_SCHEMA = {
    # "train", "val", "test", or e.g. "train_yolo_sampled".
    "split": pl.String,
    # A class, or "total" for the per-split totals row.
    "label": pl.String,
    "frames": pl.Int64,
    "frac_of_split": pl.Float64,
    "stretches": pl.Int64,
    "cameras": pl.Int64,
    "videos": pl.Int64,
    "cages": pl.Int64,
    # Only set on "total" rows: boxed frames with no behavior label.
    "labeling_hole_frames": pl.Int64,
}

TIMINGS_SCHEMA = {
    "video_id": pl.String,
    "stage": pl.String,
    "seconds": pl.Float64,
    "frames": pl.Int64,
    "gpu": pl.String,
}

SPLITS = ("train", "val", "test")
