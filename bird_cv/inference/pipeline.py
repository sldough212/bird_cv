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
from collections.abc import Iterable, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from bird_cv.classification.types import FramePrediction

from bird_cv.core.labels import Behaviors
from bird_cv.core.ids import parse_raw_video_path
from bird_cv.core.layout import InferRun
from bird_cv.core.manifest import new_manifest, write_manifest_file
from bird_cv.core.schemas import (
    DETECTION_SCHEMA,
    PREDICTION_SCHEMA,
    detection_columns,
)
from bird_cv.core.timing import StageTimer
from bird_cv.detection.types import Detection
from bird_cv.inference.cages import CageMasks, cage_frame
from bird_cv.inference.chunks import iter_chunks, stitch_overlapping
from bird_cv.inference.classify import (
    bird_crop_rgb,
    classify_runs,
    contiguous_runs,
    smoothed_box_centers,
)
from bird_cv.inference.gap_fill import cage_predictions, fill_gaps
from bird_cv.inference.protocols import (
    CageSegmenter,
    FrameClassifier,
    StreamingDetector,
)
from bird_cv.inference.registry import InferenceModels
from bird_cv.preprocessing.standardize import (
    expected_standard_frames,
    standardize_video,
    video_header,
)
from bird_cv.pipelines.config import InferConfig


def infer_video(
    raw_video_path: Path,
    run_dir: Path,
    segmenter: CageSegmenter,
    detector: StreamingDetector,
    classifier: FrameClassifier,
    cfg: InferConfig,
    timer: StageTimer,
    *,
    standard_fps: float,
    vocab: Behaviors,
    max_frames: int | None = None,
) -> pl.DataFrame:
    """Run the full pipeline on one raw video and write its outputs.

    Writes the video's standardized copy, raw detections and predictions
    where :class:`~bird_cv.core.layout.InferRun` puts them in ``run_dir``.

    Args:
        raw_video_path: Raw video at ``.../{camera_id}/{video_id}.{ext}``.
        run_dir: Output directory of the run.
        segmenter: Splits frames into cages (SAM2 for real cameras).
        detector: Detector whose trackers persist across chunks.
        classifier: Behavior classifier; only ``bbox_crop`` input is supported.
        cfg: Inference settings.
        timer: Collects one row per stage.
        standard_fps: Frame rate to standardize the video to.
        vocab: The class vocabulary predictions are checked against.
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
    out = InferRun(run_dir)
    std_path = out.standardized(camera_id, video_id)
    out_path = out.predictions(camera_id, video_id)

    expected = _expected_frames(raw_video_path, standard_fps, max_frames)
    with timer.stage(camera_id, video_id, "standardize", frames=expected):
        result = standardize_video(
            raw_video_path, std_path, fps=standard_fps, max_frames=max_frames
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {raw_video_path}: "
            f"{result.stderr.decode(errors='replace')}"
        )
    approx_frames = video_header(std_path)[1]

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
                idx: list[int] = []
                crops: list[np.ndarray] = []
                for i, frame in enumerate(frames):
                    cropped = cage_frame(
                        frame, masks.at(start + i, cage_id), shapes[cage_id]
                    )
                    if cropped is None:
                        empty[cage_id].add(start + i)
                        continue
                    idx.append(start + i)
                    crops.append(cropped)
                # Track each contiguous stretch of non-empty cage frames
                for lo, hi in contiguous_runs(idx):
                    detections[cage_id].extend(tracker.track(crops[lo:hi], idx[lo]))
    _write_detections(
        detections, camera_id, video_id, out.detections(camera_id, video_id)
    )

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
                    vocab,
                )
                for cage_id in cage_ids
            ]
        )
        _write_parquet(predictions, out_path)
    return predictions


def _expected_frames(
    video_path: Path, standard_fps: float, max_frames: int | None
) -> int:
    """Standardized frame count of a raw video from its header, for timing rows."""
    fps, count = video_header(video_path)
    if not fps:
        return 0
    return expected_standard_frames(count, fps, standard_fps, max_frames=max_frames)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    df.write_parquet(path)


def detection_rows(
    detections: Iterable[Detection], camera_id: str, video_id: str, cage_id: str
) -> list[dict[str, Any]]:
    """:data:`DETECTION_SCHEMA` rows: the ids plus each ``Detection``'s fields.

    Fields are renamed by :func:`~bird_cv.core.schemas.detection_columns`.

    Raises:
        TypeError: If ``Detection``'s fields no longer match the schema.
    """
    names = detection_columns()
    columns = ("camera_id", "video_id", "cage_id") + tuple(names.values())
    if tuple(f.name for f in fields(Detection)) != tuple(names) or columns != tuple(
        DETECTION_SCHEMA
    ):
        raise TypeError("Detection fields and DETECTION_SCHEMA have drifted apart")
    ids = {"camera_id": camera_id, "video_id": video_id, "cage_id": cage_id}
    return [{**ids, **{names[k]: v for k, v in asdict(d).items()}} for d in detections]


def _write_detections(
    detections: dict[str, list[Detection]],
    camera_id: str,
    video_id: str,
    path: Path,
) -> None:
    """Write raw tracker output, before filtering and gap fill.

    Args:
        detections: Each cage's detections, frames canonical.
        camera_id: Camera id.
        video_id: Video id.
        path: Parquet to write, with :data:`DETECTION_SCHEMA`.
    """
    rows = [
        row
        for cage_id, dets in detections.items()
        for row in detection_rows(dets, camera_id, video_id, cage_id)
    ]
    _write_parquet(pl.DataFrame(rows, schema=DETECTION_SCHEMA), path)


def _write_task_manifest(
    run: InferRun,
    task_index: int,
    config: dict[str, Any],
    inputs: dict[str, str],
) -> None:
    """Write this task's manifest so array tasks never overwrite each other."""
    write_manifest_file(
        new_manifest("infer", config=config, inputs=inputs),
        run.task_manifest(task_index),
    )


def run_infer(
    videos: Sequence[Path],
    run_dir: Path,
    models: InferenceModels,
    cfg: InferConfig,
    *,
    standard_fps: float,
    vocab: Behaviors,
    task_index: int = 0,
    num_tasks: int = 1,
    limit: int | None = None,
) -> list[Path]:
    """Run inference on this task's share of ``videos``.

    Every camera in ``videos`` is checked by the segmenter (e.g. for a SAM2
    prompt config) before any work starts. Task ``i`` of ``n`` processes
    ``videos[i::n]``; a video that fails is logged and skipped (a model that
    fails to load raises before any video). Writes this
    task's timings, manifest (config, model kinds, checkpoints, videos,
    ``limit`` and failures) and failed videos where
    :class:`~bird_cv.core.layout.InferRun` puts them in ``run_dir``.

    Args:
        videos: Raw videos at ``.../{camera_id}/{video_id}.{ext}``.
        run_dir: Output directory shared by all tasks of the run.
        models: The models config picked
            (:func:`bird_cv.inference.registry.inference_models`); loaded
            only if this task has videos.
        cfg: Inference settings.
        standard_fps: Frame rate to standardize videos to.
        vocab: The class vocabulary predictions are checked against.
        task_index: This task's index (e.g. ``SLURM_ARRAY_TASK_ID``).
        num_tasks: Number of tasks sharing ``videos``.
        limit: The ``--limit`` that selected ``videos``, recorded in the
            manifest.

    Returns:
        The videos that failed.

    Raises:
        ValueError: If ``chunk_overlap`` is not in ``[0, chunk_frames)``.
        FileNotFoundError: If the segmenter can't handle a camera (e.g. no
            SAM2 prompt config).
    """
    if not 0 <= cfg.chunk_overlap < cfg.chunk_frames:
        raise ValueError(
            f"chunk_overlap ({cfg.chunk_overlap}) must be in "
            f"[0, chunk_frames={cfg.chunk_frames})"
        )
    segmenters = models.segmenters()
    segmenters.check_cameras({parse_raw_video_path(v)[0] for v in videos})

    run = InferRun(run_dir)
    my_videos = list(videos)[task_index::num_tasks]
    config: dict[str, Any] = {
        **asdict(cfg),
        "standard_fps": standard_fps,
        "models": models.kinds(),
        "train_run_id": models.train_run_id,
        "dataset": models.dataset,
        "sam_model_id": models.context.sam_model_id,
        "task_index": task_index,
        "num_tasks": num_tasks,
        "limit": limit,
        "videos": [str(v) for v in my_videos],
    }
    inputs = models.inputs()
    run_dir.mkdir(exist_ok=True, parents=True)
    _write_task_manifest(run, task_index, config, inputs)
    if not my_videos:
        return []

    # Model loading fails the whole task, not each video
    segmenters.load()
    detector, classifier = models.load()
    timer = StageTimer()

    failed: dict[str, str] = {}
    for video in my_videos:
        camera_id, video_id = parse_raw_video_path(video)
        print(f"Inferring {camera_id}/{video_id}")
        try:
            infer_video(
                video,
                run_dir,
                segmenters.for_camera(camera_id),
                detector,
                classifier,
                cfg,
                timer,
                standard_fps=standard_fps,
                vocab=vocab,
            )
        except Exception as e:
            print(f"FAILED {camera_id}/{video_id}:\n{traceback.format_exc()}")
            failed[str(video)] = f"{type(e).__name__}: {e}"
        # Rewrite after every video so a crash keeps earlier results
        timer.write(run.timings(task_index, num_tasks))
        run.failed(task_index).write_text(json.dumps(failed, indent=2))

    _write_task_manifest(run, task_index, {**config, "failed": failed}, inputs)
    return [Path(v) for v in failed]
