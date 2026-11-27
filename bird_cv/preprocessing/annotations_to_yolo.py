import ijson
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from typing import Any, Dict, Optional
import polars as pl

from bird_cv.preprocessing.crop import crop_bird, smooth_centers
from bird_cv.core.ids import get_local_path, parse_media_path, with_raw_video_ids


def _save_cropped_bird_frame(
    frame: Any,
    frame_file: Path,
    yolo_lines: list[str],
    crop_size: int = 224,
    cx_norm: float | None = None,
    cy_norm: float | None = None,
) -> None:
    """Crop a fixed square region centered on each bird bbox and save.

    Pads with black if the crop extends beyond the frame boundary.

    Args:
        frame: BGR frame from cv2.
        frame_file: Output path for the cropped image jpg.
        yolo_lines: YOLO annotation strings for this frame.
        crop_size: Side length in pixels of the square crop.
        cx_norm: Optional smoothed x-center override (normalized 0-1).
        cy_norm: Optional smoothed y-center override (normalized 0-1).
    """
    for i, line in enumerate(yolo_lines):
        parts = line.strip().split()
        crop = crop_bird(
            frame,
            cx_norm if cx_norm is not None else float(parts[1]),
            cy_norm if cy_norm is not None else float(parts[2]),
            crop_size=crop_size,
        )

        suffix = f"_bird_{i}" if len(yolo_lines) > 1 else ""
        cv2.imwrite(str(frame_file.with_stem(frame_file.stem + suffix)), crop)


def process_item(
    item: Dict[str, Any],
    path_to_videos: Path,
    path_to_output: Path,
    path_to_guidance: Path,
    crop_size: int | None = None,
    *,
    by_cage: bool = False,
) -> None:
    """Process a single annotation item and write YOLO-format labels per frame.

    This function converts frame-level bounding box annotations from a nested
    JSON structure into YOLO-format `.txt` files, one per frame, for a single
    video. Bounding boxes are normalized using the video frame dimensions.

    The function assumes that each annotation item corresponds to exactly one
    video and that bounding box coordinates are expressed in absolute pixel
    values.

    Args:
        item: A dictionary representing one top-level annotation entry from
            the JSON file. This must include video metadata under `item["data"]`
            and annotation data under `item["annotations"]`.
        path_to_videos: Root directory containing all video files referenced
            in the annotation JSON. Unused with ``by_cage``.
        path_to_output: Directory where output files will be written. In YOLO
            mode, one image and label file is created per frame. In crop mode,
            one cropped JPEG is saved per annotated frame.
        path_to_guidance: Path to the split guidance parquet used to filter
            videos and determine train/val/test split assignment.
        crop_size: If provided, saves a square crop of this side length (in
            pixels) centered on each bird bbox instead of the full frame and
            YOLO label. Bbox centers are smoothed across frames to reduce
            YOLO jitter before cropping.
        by_cage: Match guidance rows on a ``cage_id`` column too, so each
            cage video gets its own ``target_frames`` (default: one row per
            camera video, shared by its cages). Also the canonical mode for
            aligned media: ids come from
            :func:`~bird_cv.core.ids.parse_media_path` as in ground truth,
            frame ``f`` of the cage video takes Label Studio frame ``f``
            (both canonical + 1), and it raises if the cage video isn't 15
            fps, the task has more than one non-cancelled annotation, or a
            frame has more than one box.

    Files are written to disk as a side effect.
    """
    if by_cage:
        # Same ids as ground truth; media from the export's local-files URL
        camera_id, video_id, cage = parse_media_path(item["data"]["video"])
        full_video_path = get_local_path(item["data"]["video"])
        video_path = str(full_video_path)
    else:
        # Get video path and clean it
        video_path = (
            item["data"]["video"]
            .split(f"{path_to_videos.name}/")[-1]
            .replace("%2C", ",")
        )
        full_video_path = path_to_videos / video_path

        # Where the annotation filenames are based
        video_path_base = path_to_videos.stem

        # Get the camera and video from the video_filename
        dirs = str(full_video_path).split(video_path_base)[-1]
        dirs_names = dirs.split("/")
        camera_id = dirs_names[1]
        video_id = dirs_names[2]
        cage = full_video_path.stem
        # Remove optional extension from the video_id
        if not (path_to_videos / camera_id / video_id).is_dir():
            video_id = str(Path(video_id).stem)

    # Load in the guidance
    video_guidance = with_raw_video_ids(pl.read_parquet(path_to_guidance))

    video_guidance = video_guidance.filter(
        (pl.col("camera_id") == camera_id) & (pl.col("video_id") == video_id)
    )
    if by_cage:
        video_guidance = video_guidance.filter(pl.col("cage_id") == cage)

    print(f"Working on {video_path}")
    if video_guidance.is_empty():
        print(f"Video {video_path} not in target")
        return

    # Determine the split of the video based on guidance and redefine output paths
    if crop_size:
        split = "train"
        path_to_output_frames = (
            path_to_output / "images" / split / camera_id / video_id / cage
        )
        path_to_output_frames.mkdir(exist_ok=True, parents=True)
    else:
        split = video_guidance.select("split").item()
        path_to_output_frames = path_to_output / "images" / split
        path_to_output_labels = path_to_output / "labels" / split

    # Collect annotations per frame
    frame_annotations: Dict[int, list[str]] = defaultdict(list)

    annotations = item["annotations"]
    if by_cage:
        annotations = [a for a in annotations if not a.get("was_cancelled")]
        if len(annotations) > 1:
            raise ValueError(
                f"{video_path}: {len(annotations)} non-cancelled annotations; "
                "by_cage needs exactly one"
            )
    for ann in annotations:
        for result in ann["result"]:
            label: Optional[str] = (  # noqa
                result["value"]["labels"][0] if result["value"].get("labels") else None
            )

            # Currently a single-class setup (e.g., bird = 0)
            class_id: int = 0

            for seq in result["value"].get("sequence", []):
                frame_num = int(seq["frame"])

                # These are given as percentages of the entire frame
                x = float(seq["x"])
                y = float(seq["y"])
                w = float(seq["width"])
                h = float(seq["height"])

                # Normalize to YOLO format
                x_center: float = (x + w / 2) / 100
                y_center: float = (y + h / 2) / 100
                w_norm: float = w / 100
                h_norm: float = h / 100

                yolo_line: str = (
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{w_norm:.6f} "
                    f"{h_norm:.6f}"
                )

                frame_annotations[frame_num].append(yolo_line)

    target_frames = video_guidance.select("target_frames").to_numpy().squeeze().tolist()
    ls_fps = video_guidance.select("fps").item()

    # Open video to extract frames
    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if by_cage:
        multi = sorted(f for f, lines in frame_annotations.items() if len(lines) > 1)
        if multi:
            cap.release()
            raise ValueError(
                f"{video_path}: frames {multi[:10]} have more than one box"
            )
        if abs(fps - ls_fps) > 0.01 * ls_fps:
            cap.release()
            raise ValueError(
                f"{full_video_path}: {fps} fps; by_cage needs the guidance's "
                f"{ls_fps} fps"
            )

    # Smooth bbox centers across target frames to reduce YOLO jitter
    smoothed_centers: dict[int, tuple[float, float]] = {}
    if crop_size:
        # Same video-to-annotation frame mapping as the extraction loop below
        annotated = [
            (f, af)
            for f in sorted(target_frames)
            if frame_annotations[af := f if by_cage else int(f * ls_fps / fps)]
        ]
        if len(annotated) >= 2:
            cx_seq = np.array(
                [float(frame_annotations[af][0].split()[1]) for _, af in annotated]
            )
            cy_seq = np.array(
                [float(frame_annotations[af][0].split()[2]) for _, af in annotated]
            )
            cx_smooth, cy_smooth = smooth_centers(cx_seq, cy_seq, window=5)
            smoothed_centers = {
                vf: (cx_smooth[i], cy_smooth[i]) for i, (vf, _) in enumerate(annotated)
            }

    # Extract that image frame and save
    frame_num = 1
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_num not in target_frames:
            frame_num += 1
            continue

        # Need to inverse the fps correction to get the correct annotation frame
        # by_cage: aligned media, video frame = Label Studio frame (canonical
        # + 1); the rescale truncates to F - 1 for fps slightly above 15
        frame_num_ann = frame_num if by_cage else int(frame_num * ls_fps / fps)
        lines = frame_annotations[frame_num_ann]

        if crop_size:
            frame_file = path_to_output_frames / f"{frame_num:05d}.jpg"
            center = smoothed_centers.get(frame_num)
            _save_cropped_bird_frame(
                frame,
                frame_file,
                lines,
                crop_size=crop_size,
                cx_norm=center[0] if center else None,
                cy_norm=center[1] if center else None,
            )
        else:
            stem = f"{camera_id}_{video_id}_{cage}_{frame_num:05d}"
            frame_file = path_to_output_frames / f"{stem}.jpg"
            cv2.imwrite(str(frame_file), frame)
            label_file = path_to_output_labels / f"{stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(lines))

        frame_num += 1

    cap.release()


def stream_annotations_to_yolo(
    path_to_videos: Path,
    path_to_annotations: Path,
    path_to_output: Path,
    path_to_guidance: Path,
    processes: int = 4,
    crop_size: int | None = None,
    *,
    by_cage: bool = False,
) -> None:
    """Stream a large JSON annotation file and convert it to YOLO format.

    This function streams a potentially large JSON annotation file using
    `ijson` and converts all video annotations into YOLO-format `.txt` files.
    Processing can be done either serially or in parallel using multiprocessing.

    Args:
        path_to_videos: Root directory containing all referenced video files.
            Unused with ``by_cage``.
        path_to_annotations: Path to the JSON annotation file to be processed.
        path_to_output: Output directory where YOLO `.txt` files will be written.
        processes: Number of worker processes to use. If set to 1, processing
            runs serially (useful for testing and debugging). Values greater
            than 1 enable multiprocessing.
        crop_size: If provided, save bird crops instead of YOLO frames and
            labels (see :func:`process_item`).
        by_cage: Guidance has one row per cage video (see :func:`process_item`).

    Returns:
        None. YOLO annotation files are written to disk as a side effect.
    """
    # Create direcotries for training
    path_to_output.mkdir(exist_ok=True, parents=True)
    splits = ["train", "val", "test"]
    for split in splits:
        (path_to_output / "images" / split).mkdir(exist_ok=True, parents=True)
        (path_to_output / "labels" / split).mkdir(exist_ok=True, parents=True)

    with open(path_to_annotations, "rb") as f:
        items = ijson.items(f, "item")  # generator over top-level items

        if processes == 1:
            # Serial execution (recommended for unit tests)
            for item in items:
                process_item(
                    item=item,
                    path_to_videos=path_to_videos,
                    path_to_output=path_to_output,
                    path_to_guidance=path_to_guidance,
                    crop_size=crop_size,
                    by_cage=by_cage,
                )
        else:
            # Parallel execution
            worker_func = partial(
                process_item,
                path_to_videos=path_to_videos,
                path_to_output=path_to_output,
                path_to_guidance=path_to_guidance,
                crop_size=crop_size,
                by_cage=by_cage,
            )

            with Pool(processes=processes) as pool:
                for _ in tqdm(
                    pool.imap(worker_func, items, chunksize=1),
                    desc="Processing videos",
                ):
                    pass
