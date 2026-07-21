from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import ijson
import numpy as np
import polars as pl
from tqdm import tqdm


def _extract_video_frames(video_path: Path) -> List[np.ndarray]:
    """Read every frame of a video into memory, in order.

    Args:
        video_path: Path to the video file.

    Returns:
        List of BGR frames as read by OpenCV, in playback order.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _center_window(
    start: int, end: int, num_frames: int, total_frames: int
) -> Tuple[int, int]:
    """Symmetrically expand a ``[start, end)`` range to length ``num_frames``.

    Extends into real neighboring frames from the source video on both
    sides so the labeled segment ends up centered in the window, rather
    than pinned to one edge. If one side runs out of room against a video
    boundary, the remaining deficit is shifted to the other side. If the
    video itself has fewer than ``num_frames`` frames, the returned range
    simply covers the whole video (the caller pads the rest).

    Args:
        start: Inclusive start index (0-based) of the labeled segment.
        end: Exclusive end index (0-based) of the labeled segment.
        num_frames: Desired window length.
        total_frames: Number of frames available in the source video.

    Returns:
        ``(new_start, new_end)``, clamped to ``[0, total_frames)``.
    """
    deficit = num_frames - (end - start)
    left_pad = deficit // 2
    right_pad = deficit - left_pad

    new_start = start - left_pad
    new_end = end + right_pad

    if new_start < 0:
        new_end += -new_start
        new_start = 0
    if new_end > total_frames:
        new_start -= new_end - total_frames
        new_end = total_frames

    return max(new_start, 0), new_end


def load_split_guidance(path_to_guidance: Path) -> Dict[Tuple[str, str], str]:
    """Load a split guidance parquet and index it by (camera_id, video_id).

    Mirrors the matching logic used in ``annotations_to_yolo.process_item``:
    each guidance record's ``video_path`` is split on "/", with the
    second-to-last segment (comma-decoded) taken as the camera id and the
    last segment (extension stripped) taken as the video id.

    Args:
        path_to_guidance: Path to a ``split_guidance.parquet`` file (e.g.
            produced by ``get_split_guidance.split_camera_data`` or
            ``segment_videos.split_guidance_within_camera``), containing at
            least ``video_path`` (path to the *source* camera video, e.g.
            ``"I20%2CI38/32.mp4"``) and ``split`` columns.

    Returns:
        Mapping from ``(camera_id, video_id)`` to split name.
    """
    guidance_df = pl.read_parquet(path_to_guidance).with_columns(
        video_id=pl.col("video_path")
        .str.split("/")
        .list.last()
        .str.replace(r"\.[^.]+$", ""),
        camera_id=pl.col("video_path")
        .str.split("/")
        .list.get(-2)
        .str.replace_all("%2C", ","),
    )

    guidance: Dict[Tuple[str, str], str] = {}
    for camera_id, video_id, split in guidance_df.select(
        "camera_id", "video_id", "split"
    ).iter_rows():
        guidance[(camera_id, video_id)] = split

    return guidance


def process_item(
    item: Dict[str, Any],
    path_to_videos: Path,
    path_to_output: Path,
    guidance: Dict[Tuple[str, str], str],
    num_frames: int = 16,
    stride: int = 8,
) -> None:
    """Convert one Label Studio behavior task into VideoMAE training clips.

    Each task corresponds to a single cropped per-cage video. Every
    non-cancelled annotation on the task carries a list of timeline-label
    results, each giving a behavior label and one or more inclusive
    ``{start, end}`` frame ranges (1-indexed, per Label Studio's video
    timeline convention). For each range, a sliding window of ``num_frames``
    is stepped across the available frames (advancing by ``stride``); ranges
    shorter than ``num_frames`` are centered and padded up to ``num_frames``
    — preferring real neighboring frames from the source video on both
    sides, and only repeating the first/last frame if the video itself is
    shorter than ``num_frames`` — so short behaviors still produce one clip
    with the labeled action centered in it.

    Output structure::

        path_to_output/
          {split}/
            {label}/
              {camera_id}_{video_id}_{cage}_a{annotation_id}_r{result_idx}_clip{idx}/
                00000.jpg
                00001.jpg
                ...

    Args:
        item: A single top-level task from the behavior annotation JSON,
            with video metadata under ``item["data"]["video"]`` and human
            annotations under ``item["annotations"]``.
        path_to_videos: Root directory containing the cropped per-cage
            videos, structured as ``{camera_id}/{video_id}/{cage_id}.mp4``.
        path_to_output: Directory where clip frames will be written.
        guidance: Split guidance mapping produced by :func:`load_split_guidance`.
        num_frames: Number of frames per clip. Must match what
            :func:`train_video_model.train_video_model` is configured with.
            Defaults to 16.
        stride: Number of frames to advance the sliding window each step.
            Defaults to 8.

    Returns:
        None. Clip frames are written to disk as a side effect.
    """
    video_field = item["data"]["video"]
    video_path = video_field.split(f"{path_to_videos.name}/")[-1].replace("%2C", ",")
    full_video_path = path_to_videos / video_path

    video_path_base = path_to_videos.stem
    dirs = str(full_video_path).split(video_path_base)[-1]
    dirs_names = dirs.split("/")
    camera_id = dirs_names[1]
    video_id = dirs_names[2]
    cage = full_video_path.stem

    split = guidance.get((camera_id, video_id))
    if split is None:
        print(f"Video {video_path} not in split guidance, skipping.")
        return

    if not full_video_path.exists():
        print(f"Video file not found: {full_video_path}, skipping.")
        return

    frames = _extract_video_frames(full_video_path)
    if not frames:
        print(f"No frames read from {full_video_path}, skipping.")
        return

    for annotation in item.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue

        annotation_id = annotation.get("id")

        for result_idx, result in enumerate(annotation.get("result", [])):
            if result.get("type") != "timelinelabels":
                continue

            value = result["value"]
            labels = value.get("timelinelabels")
            if not labels:
                continue

            label = labels[0]
            label_dir = label.replace("/", "_").replace(" ", "_").lower()

            for frame_range in value.get("ranges", []):
                start = max(int(frame_range["start"]) - 1, 0)
                end = min(int(frame_range["end"]), len(frames))

                if end <= start:
                    print(
                        f"Segment {camera_id}/{video_id}/{cage} annotation "
                        f"{annotation_id} result {result_idx} has no frames, "
                        "skipping."
                    )
                    continue

                if end - start < num_frames:
                    # Prefer expanding into real neighboring frames from the
                    # source video on both sides, so the labeled behavior
                    # ends up centered in the clip.
                    start, end = _center_window(start, end, num_frames, len(frames))

                candidates = frames[start:end]

                if len(candidates) < num_frames:
                    # Video itself is shorter than num_frames even at full
                    # length — pad the remainder symmetrically by repeating
                    # the first/last frame.
                    remaining = num_frames - len(candidates)
                    left_pad = remaining // 2
                    right_pad = remaining - left_pad
                    candidates = (
                        [candidates[0]] * left_pad
                        + candidates
                        + [candidates[-1]] * right_pad
                    )

                starts = range(0, len(candidates) - num_frames + 1, stride)
                for clip_idx, clip_start in enumerate(starts):
                    window = candidates[clip_start : clip_start + num_frames]
                    clip_dir = (
                        path_to_output
                        / split
                        / label_dir
                        / (
                            f"{camera_id}_{video_id}_{cage}_a{annotation_id}"
                            f"_r{result_idx}_clip{clip_idx:03d}"
                        )
                    )
                    clip_dir.mkdir(parents=True, exist_ok=True)
                    for out_idx, frame in enumerate(window):
                        cv2.imwrite(str(clip_dir / f"{out_idx:05d}.jpg"), frame)


def stream_annotations_to_mae(
    path_to_videos: Path,
    path_to_annotations: Path,
    path_to_guidance: Path,
    path_to_output: Path,
    num_frames: int = 16,
    stride: int = 8,
    processes: int = 4,
) -> None:
    """Stream a behavior annotation JSON and build VideoMAE train/val/test clips.

    Streams a potentially large Label Studio behavior export using ``ijson``
    and converts every timeline-label annotation into fixed-length frame
    clips, split into ``train``/``val``/``test`` per ``path_to_guidance``,
    ready for :class:`bird_cv.classification.train_video_model.BehaviorClipDataset`.

    Args:
        path_to_videos: Root directory containing the cropped per-cage
            videos, structured as ``{camera_id}/{video_id}/{cage_id}.mp4``.
        path_to_annotations: Path to the behavior annotation JSON exported
            from Label Studio.
        path_to_guidance: Path to the split guidance parquet (see
            :func:`load_split_guidance`).
        path_to_output: Output directory where clip frames will be written.
        num_frames: Number of frames per clip. Defaults to 16.
        stride: Number of frames to advance the sliding window each step.
            Defaults to 8.
        processes: Number of worker processes to use. If set to 1, processing
            runs serially (useful for testing and debugging). Values greater
            than 1 enable multiprocessing.

    Returns:
        None. Clip frames are written to disk as a side effect.
    """
    path_to_output.mkdir(exist_ok=True, parents=True)
    guidance = load_split_guidance(path_to_guidance)

    with open(path_to_annotations, "rb") as f:
        items = ijson.items(f, "item")

        if processes == 1:
            # Serial execution (recommended for unit tests)
            for item in items:
                process_item(
                    item=item,
                    path_to_videos=path_to_videos,
                    path_to_output=path_to_output,
                    guidance=guidance,
                    num_frames=num_frames,
                    stride=stride,
                )
        else:
            # Parallel execution
            worker_func = partial(
                process_item,
                path_to_videos=path_to_videos,
                path_to_output=path_to_output,
                guidance=guidance,
                num_frames=num_frames,
                stride=stride,
            )

            with Pool(processes=processes) as pool:
                for _ in tqdm(
                    pool.imap(worker_func, items, chunksize=1),
                    desc="Processing behavior clips",
                ):
                    pass
