"""Extract fixed-length frame clips for each behavior in the behavior index."""

from pathlib import Path

import polars as pl
import shutil


def extract_behavior_clips(
    behavior_index_path: Path,
    cropped_frames_path: Path,
    output_path: Path,
    num_frames: int = 16,
    stride: int = 8,
) -> None:
    """Extract fixed-length sliding window clips from cropped cage images for VideoMAE.

    For each behavior in the behavior index, slides a window of ``num_frames`` across
    the available cropped images within ``[frame_begin, frame_end]``, advancing by
    ``stride`` frames each step. Each window is saved as a separate numbered clip.
    Trailing windows shorter than ``num_frames`` are dropped. Raises if fewer than
    ``num_frames`` candidate frames exist for a behavior.

    Output structure::

        output_path/
          {split}/
            {label}/
              {track_id}_{cage_id}/
                00000.jpg
                00001.jpg
                ...

    Args:
        behavior_index_path: Path to the behavior index parquet produced by
            :func:`run_crop_yolo`.
        cropped_frames_path: Root of the per-cage cropped YOLO image output
            (``yolo_crop/images/``).
        output_path: Directory where clip frames will be written.
        num_frames: Number of frames to sample per clip. Defaults to 16.
    """
    behavior_index = pl.read_parquet(behavior_index_path)

    for row in behavior_index.iter_rows(named=True):
        camera_id = row["camera_id"]
        video_id = row["video_id"]
        cage_id = str(row["cage_id"])
        label = row["label"]
        split = row["split"]
        track_id = row["track_id"]
        frame_begin = row["frame_begin"]
        frame_end = row["frame_end"]

        yolo_camera_id = camera_id.replace(",", "%2C")

        # Locate candidate frames on disk
        if split == "test":
            search_dir = cropped_frames_path / split / camera_id / video_id / cage_id
        else:
            search_dir = cropped_frames_path / split

        pattern = f"{yolo_camera_id}.{video_id}_frame_*_cage_{cage_id}.jpg"
        candidates = sorted(search_dir.glob(pattern))

        # Filter to frames within [frame_begin, frame_end]
        def _frame_num(p: Path) -> int:
            # filename: {camera}.{video}_frame_{num:05d}_cage_{cage}.jpg
            return int(p.stem.split("_frame_")[1].split("_cage_")[0])

        candidates = [
            p for p in candidates if frame_begin <= _frame_num(p) <= frame_end
        ]

        if not candidates:
            print(
                f"No frames found for track {track_id} cage {cage_id} "
                f"({camera_id}/{video_id} frames {frame_begin}-{frame_end}), skipping."
            )
            continue

        if len(candidates) < num_frames:
            raise ValueError(
                f"Track {track_id} cage {cage_id} ({camera_id}/{video_id}) has only "
                f"{len(candidates)} candidate frames, expected at least {num_frames}. "
                "Check that behavior buffering in get_label_tables is configured correctly."
            )

        # Sanitize label for use as a directory name
        label_dir = label.replace("/", "_").replace(" ", "_").lower()

        # Slide a window of num_frames; drop any trailing window that would be short
        starts = range(0, len(candidates) - num_frames + 1, stride)
        for clip_idx, start in enumerate(starts):
            window = candidates[start : start + num_frames]
            clip_dir = (
                output_path
                / split
                / label_dir
                / f"{track_id}_{cage_id}_clip{clip_idx:03d}"
            )
            clip_dir.mkdir(parents=True, exist_ok=True)
            for out_idx, src in enumerate(window):
                shutil.copy(src, clip_dir / f"{out_idx:05d}.jpg")
