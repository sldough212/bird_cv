"""End-to-end inference: raw video to one :data:`PREDICTION_SCHEMA` parquet.

Per video, in three passes over the standardized video, ``chunk_frames``
canonical frames at a time:

1. ``sam2``: cage keyframe masks per chunk.
2. ``detect``: each cage's frames (cropped, masked, padded to one size per
   cage) through its own tracker, whose state carries across chunks.
3. ``classify``: after confidence filtering and gap fill over the whole
   video, bird crops of boxed frames through the classifier, run by run,
   with chunk windows overlapping by ``chunk_overlap`` frames and stitched.

Output is keyed by ``cage_id`` and canonical frame, one row per cage per
frame; frames with no usable box are ``no_bird``.
"""

import json
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import msgspec
import numpy as np
import polars as pl
import torch
from sam2.build_sam import build_sam2_video_predictor_hf

from bird_cv.classification.predict_video_model import VideoMAEClassifier
from bird_cv.classification.types import FramePrediction

from bird_cv.core.frames import STANDARD_FPS
from bird_cv.core.ids import parse_raw_video_path
from bird_cv.core.manifest import new_manifest
from bird_cv.core.schemas import PREDICTION_SCHEMA
from bird_cv.core.timing import StageTimer
from bird_cv.detection.track import YoloDetector
from bird_cv.detection.types import Detection
from bird_cv.inference.cages import (
    CageMasks,
    Sam2CageSegmenter,
    cage_frame,
    check_sam2_configs,
    sam2_config_path,
    sam2_reference_frame,
)
from bird_cv.inference.chunks import iter_chunks, stitch_overlapping
from bird_cv.inference.classify import (
    bird_crop_rgb,
    classify_runs,
    smoothed_box_centers,
)
from bird_cv.inference.gap_fill import cage_predictions, fill_gaps
from bird_cv.inference.protocols import (
    CageSegmenter,
    FrameClassifier,
    StreamingDetector,
)
from bird_cv.preprocessing.standardize import standardize_video
from bird_cv.pipelines.config import InferConfig

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

DETECTION_SCHEMA = {
    "camera_id": pl.String,
    "video_id": pl.String,
    "cage_id": pl.String,
    "frame": pl.Int64,
    "track_id": pl.Int64,
    "bbox_left": pl.Float64,
    "bbox_top": pl.Float64,
    "bbox_width": pl.Float64,
    "bbox_height": pl.Float64,
    "det_conf": pl.Float64,
}
"""Raw tracker output before confidence filter and gap fill (``detections/``)."""


def infer_video(
    raw_video_path: Path,
    run_dir: Path,
    segmenter: CageSegmenter,
    detector: StreamingDetector,
    classifier: FrameClassifier,
    cfg: InferConfig,
    timer: StageTimer,
    *,
    max_frames: int | None = None,
) -> pl.DataFrame:
    """Run the full pipeline on one raw video and write its outputs.

    Writes, under ``run_dir``: ``standardized/{camera_id}/{video_id}.mp4``,
    ``detections/{camera_id}/{video_id}.parquet`` (raw tracks) and
    ``predictions/{camera_id}/{video_id}.parquet``.

    Args:
        raw_video_path: Raw video at ``.../{camera_id}/{video_id}.{ext}``.
        run_dir: Output directory of the run.
        segmenter: Splits frames into cages (SAM2 for real cameras).
        detector: Detector whose trackers persist across chunks.
        classifier: Behavior classifier; only ``bbox_crop`` input is supported.
        cfg: Inference settings.
        timer: Collects one row per stage.
        max_frames: Optionally truncate the standardized video.

    Returns:
        The predictions, :data:`PREDICTION_SCHEMA`, sorted by cage and frame
        (empty if the standardized video has no frames).

    Raises:
        NotImplementedError: For classifiers taking ``cage`` input.
        RuntimeError: If ffmpeg fails to standardize the video.
    """
    if classifier.input_type != "bbox_crop":
        raise NotImplementedError(
            f"Classifier input {classifier.input_type!r} is not supported yet"
        )
    camera_id, video_id = parse_raw_video_path(raw_video_path)
    std_path = run_dir / "standardized" / camera_id / f"{video_id}.mp4"
    out_path = run_dir / "predictions" / camera_id / f"{video_id}.parquet"

    expected = _expected_frames(raw_video_path)
    if max_frames is not None:
        expected = min(expected, max_frames)
    with timer.stage(camera_id, video_id, "standardize", frames=expected):
        result = standardize_video(
            raw_video_path, std_path, fps=STANDARD_FPS, max_frames=max_frames
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {raw_video_path}: "
            f"{result.stderr.decode(errors='replace')}"
        )
    cap = cv2.VideoCapture(str(std_path))
    approx_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Pass 1: cage masks
    masks = CageMasks()
    n_frames = 0
    with timer.stage(camera_id, video_id, "sam2", frames=approx_frames):
        for start, frames in iter_chunks(std_path, cfg.chunk_frames):
            masks.add(segmenter.segment_chunk(frames, start))
            n_frames = start + len(frames)
    if n_frames == 0:
        print(f"{raw_video_path}: standardized video has no frames")
        predictions = pl.DataFrame(schema=PREDICTION_SCHEMA)
        _write_parquet(predictions, out_path)
        return predictions
    shapes = masks.cage_frame_shapes()
    cage_ids = masks.cage_ids

    # Pass 2: tracking, one tracker per cage
    trackers = {c: detector.frame_tracker() for c in shapes}
    detections: dict[str, list[Detection]] = {c: [] for c in cage_ids}
    empty: dict[str, set[int]] = {c: set() for c in cage_ids}
    with timer.stage(camera_id, video_id, "detect", frames=n_frames):
        for start, frames in iter_chunks(std_path, cfg.chunk_frames):
            for cage_id, tracker in trackers.items():
                cage_frames: list[tuple[int, np.ndarray]] = []
                for i, frame in enumerate(frames):
                    cropped = cage_frame(
                        frame, masks.at(start + i, cage_id), shapes[cage_id]
                    )
                    if cropped is None:
                        empty[cage_id].add(start + i)
                        continue
                    cage_frames.append((start + i, cropped))
                # Track each contiguous stretch of non-empty cage frames
                for run_start, run in _split_runs(cage_frames):
                    detections[cage_id].extend(tracker.track(run, run_start))
    _write_detections(detections, camera_id, video_id, run_dir / "detections")

    boxes = {
        c: fill_gaps(
            detections[c],
            n_frames,
            det_conf=cfg.det_conf,
            max_gap=cfg.gap_fill_frames,
            empty_frames=empty[c] if c in shapes else range(n_frames),
        )
        for c in cage_ids
    }
    centers = {c: smoothed_box_centers(boxes[c]) for c in cage_ids}

    # Pass 3: classification over boxed frames, overlapping chunk windows
    windows: dict[str, list[tuple[int, int, dict[int, FramePrediction]]]] = {
        c: [] for c in cage_ids
    }
    with timer.stage(camera_id, video_id, "classify", frames=n_frames):
        for start, frames in iter_chunks(std_path, cfg.chunk_frames, cfg.chunk_overlap):
            end = start + len(frames)
            for cage_id in cage_ids:
                boxed = [f for f in range(start, end) if f in centers[cage_id]]
                crops = []
                for f in boxed:
                    cropped = cage_frame(
                        frames[f - start], masks.at(f, cage_id), shapes[cage_id]
                    )
                    assert cropped is not None  # empty cage-frames have no box
                    crops.append(
                        bird_crop_rgb(cropped, centers[cage_id][f], cfg.crop_size)
                    )
                preds = classify_runs(boxed, crops, classifier) if boxed else {}
                windows[cage_id].append((start, end, preds))

    with timer.stage(camera_id, video_id, "write", frames=n_frames):
        predictions = pl.concat(
            [
                cage_predictions(
                    camera_id,
                    video_id,
                    cage_id,
                    boxes[cage_id],
                    stitch_overlapping(windows[cage_id]),
                )
                for cage_id in cage_ids
            ]
        )
        _write_parquet(predictions, out_path)
    return predictions


def _expected_frames(video_path: Path) -> int:
    """Approximate standardized frame count of a raw video, for timing rows."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return round(count * STANDARD_FPS / fps) if fps else 0


def _split_runs(
    frames: Sequence[tuple[int, np.ndarray]],
) -> list[tuple[int, list[np.ndarray]]]:
    """Group ``(canonical_frame, frame)`` pairs into runs of consecutive frames.

    Returns:
        ``(first canonical frame, frames)`` per run, in order.
    """
    runs: list[tuple[int, list[np.ndarray]]] = []
    for frame_idx, frame in frames:
        if runs and runs[-1][0] + len(runs[-1][1]) == frame_idx:
            runs[-1][1].append(frame)
        else:
            runs.append((frame_idx, [frame]))
    return runs


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(path)


def _write_detections(
    detections: dict[str, list[Detection]],
    camera_id: str,
    video_id: str,
    output_dir: Path,
) -> None:
    """Write raw tracker output, before filtering and gap fill.

    Args:
        detections: Each cage's detections, frames canonical.
        camera_id: Camera id.
        video_id: Video id.
        output_dir: Root; writes ``{camera_id}/{video_id}.parquet`` under it
            with :data:`DETECTION_SCHEMA`.
    """
    rows = [
        {
            "camera_id": camera_id,
            "video_id": video_id,
            "cage_id": cage_id,
            "frame": d.frame,
            "track_id": d.track_id,
            "bbox_left": d.left,
            "bbox_top": d.top,
            "bbox_width": d.width,
            "bbox_height": d.height,
            "det_conf": d.conf,
        }
        for cage_id, dets in detections.items()
        for d in dets
    ]
    _write_parquet(
        pl.DataFrame(rows, schema=DETECTION_SCHEMA),
        output_dir / camera_id / f"{video_id}.parquet",
    )


def find_raw_videos(
    raw_videos_path: Path, cameras: Sequence[str] | None = None
) -> list[Path]:
    """Raw videos under ``{raw_videos_path}/{camera_id}/``, sorted, optionally filtered."""
    videos = sorted(
        p
        for p in raw_videos_path.glob("*/*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if cameras is not None:
        videos = [v for v in videos if parse_raw_video_path(v)[0] in cameras]
    return videos


def _write_task_manifest(
    run_dir: Path,
    task_index: int,
    config: dict[str, Any],
    inputs: dict[str, str],
) -> None:
    """Write ``manifest_task{i}.json`` so array tasks never overwrite each other."""
    manifest = new_manifest("infer", config=config, inputs=inputs)
    path = run_dir / f"manifest_task{task_index:03d}.json"
    path.write_bytes(msgspec.json.format(msgspec.json.encode(manifest)))


def run_infer(
    videos: Sequence[Path],
    run_dir: Path,
    segmentation_configs_path: Path,
    sam_model_id: str,
    yolo_path: Path,
    tracker_path: Path,
    videomae_path: Path,
    cfg: InferConfig,
    *,
    task_index: int = 0,
    num_tasks: int = 1,
    limit: int | None = None,
) -> list[Path]:
    """Run inference on this task's share of ``videos``.

    Every camera in ``videos`` is checked for a SAM2 prompt config before
    any work starts. Task ``i`` of ``n`` processes ``videos[i::n]``; a video
    that fails is logged and skipped. Writes into ``run_dir``:
    ``timings.parquet`` (``timings_task{i:03d}.parquet`` when ``n > 1``),
    ``manifest_task{i:03d}.json`` (config, this task's videos, ``limit`` and
    failures) and ``failed_task{i:03d}.json`` (failed videos and errors).

    Args:
        videos: Raw videos at ``.../{camera_id}/{video_id}.{ext}``.
        run_dir: Output directory shared by all tasks of the run.
        segmentation_configs_path: Root holding SAM2 ``configs/`` and ``frames/``.
        sam_model_id: Hugging Face id of the SAM2 checkpoint.
        yolo_path: YOLO weights.
        tracker_path: Tracker config.
        videomae_path: VideoMAE checkpoint directory.
        cfg: Inference settings.
        task_index: This task's index (e.g. ``SLURM_ARRAY_TASK_ID``).
        num_tasks: Number of tasks sharing ``videos``.
        limit: The ``--limit`` that selected ``videos``, recorded in the
            manifest.

    Returns:
        The videos that failed.

    Raises:
        ValueError: If ``chunk_overlap`` is not in ``[0, chunk_frames)``.
        FileNotFoundError: If a camera lacks a SAM2 prompt config.
    """
    if not 0 <= cfg.chunk_overlap < cfg.chunk_frames:
        raise ValueError(
            f"chunk_overlap ({cfg.chunk_overlap}) must be in "
            f"[0, chunk_frames={cfg.chunk_frames})"
        )
    check_sam2_configs(
        segmentation_configs_path, {parse_raw_video_path(v)[0] for v in videos}
    )

    my_videos = list(videos)[task_index::num_tasks]
    config: dict[str, Any] = {
        **asdict(cfg),
        "sam_model_id": sam_model_id,
        "task_index": task_index,
        "num_tasks": num_tasks,
        "limit": limit,
        "videos": [str(v) for v in my_videos],
    }
    inputs = {
        "yolo": str(yolo_path),
        "tracker": str(tracker_path),
        "videomae": str(videomae_path),
        "segmentation_configs": str(segmentation_configs_path),
    }
    run_dir.mkdir(exist_ok=True, parents=True)
    _write_task_manifest(run_dir, task_index, config, inputs)
    if not my_videos:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor_hf(sam_model_id, device=device)
    detector = YoloDetector(model_path=yolo_path, tracker_path=tracker_path)
    classifier = VideoMAEClassifier(model_path=str(videomae_path))
    timer = StageTimer()
    timings_name = (
        "timings.parquet" if num_tasks == 1 else f"timings_task{task_index:03d}.parquet"
    )

    failed: dict[str, str] = {}
    for video in my_videos:
        camera_id, video_id = parse_raw_video_path(video)
        print(f"Inferring {camera_id}/{video_id}")
        try:
            segmenter = Sam2CageSegmenter(
                predictor=predictor,
                sam_model_id=sam_model_id,
                config_path=sam2_config_path(segmentation_configs_path, camera_id),
                reference_frame_path=sam2_reference_frame(
                    segmentation_configs_path, camera_id
                ),
                device=device,
            )
            infer_video(video, run_dir, segmenter, detector, classifier, cfg, timer)
        except Exception as e:
            print(f"FAILED {camera_id}/{video_id}:\n{traceback.format_exc()}")
            failed[str(video)] = f"{type(e).__name__}: {e}"
        # Rewrite after every video so a crash keeps earlier results
        timer.write(run_dir / timings_name)
        (run_dir / f"failed_task{task_index:03d}.json").write_text(
            json.dumps(failed, indent=2)
        )

    _write_task_manifest(run_dir, task_index, {**config, "failed": failed}, inputs)
    return [Path(v) for v in failed]
