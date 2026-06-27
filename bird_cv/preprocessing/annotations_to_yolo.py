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
    frame_h, frame_w = frame.shape[:2]
    half = crop_size // 2

    for i, line in enumerate(yolo_lines):
        parts = line.strip().split()
        cx = int((cx_norm if cx_norm is not None else float(parts[1])) * frame_w)
        cy = int((cy_norm if cy_norm is not None else float(parts[2])) * frame_h)

        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - frame_w)
        pad_bottom = max(0, y2 - frame_h)

        crop = frame[max(0, y1) : min(frame_h, y2), max(0, x1) : min(frame_w, x2)]

        if pad_left or pad_top or pad_right or pad_bottom:
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        suffix = f"_bird_{i}" if len(yolo_lines) > 1 else ""
        cv2.imwrite(str(frame_file.with_stem(frame_file.stem + suffix)), crop)


def process_item(
    item: Dict[str, Any],
    path_to_videos: Path,
    path_to_output: Path,
    path_to_guidance: Path,
    crop_size: int | None = None,
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
            in the annotation JSON.
        path_to_output: Directory where YOLO `.txt` annotation files will be
            written. One file is created per video frame.

    Returns:
        None. Annotation files are written to disk as a side effect.
    """
    # Get video path and clean it
    video_path = (
        item["data"]["video"].split("2021_bunting_clips/")[-1].replace("%2C", ",")
    )
    full_video_path = path_to_videos / video_path

    # Get a unique identifier for each video
    video_filename = video_path.replace("/", ".").replace(",", "%2C")

    # Where the annotation filenames are based
    video_path_base = path_to_videos.stem

    # Get the camera and video from the video_filename
    dirs = str(full_video_path).split(video_path_base)[-1]
    dirs_names = dirs.split("/")
    camera_id = dirs_names[1]
    video_id = dirs_names[2]
    # Remove optional extension from the video_id
    if not (path_to_videos / camera_id / video_id).is_dir():
        video_id = str(Path(video_id).stem)

    # Load in the guidance
    guidance = pl.read_parquet(path_to_guidance)
    video_guidance = guidance.with_columns(
        video_id=pl.col("video_path")
        .str.split("/")
        .list.last()
        .str.replace(r"\.[^.]+$", ""),
        camera_id=pl.col("video_path")
        .str.split("/")
        .list.get(-2)
        .str.replace_all("%2C", ","),
    )

    video_guidance = video_guidance.filter(
        (pl.col("camera_id") == camera_id) & (pl.col("video_id") == video_id)
    )

    print(f"Working on {video_path}")
    if video_guidance.is_empty():
        print(f"Video {video_path} not in target")
        return

    # Determine the split of the video based on guidance and redefine output paths
    if crop_size:
        split = "train"
    else:
        split = video_guidance.select("split").item()

    path_to_output_frames = path_to_output / "images" / split
    path_to_output_labels = path_to_output / "labels" / split

    # Collect annotations per frame
    frame_annotations: Dict[int, list[str]] = defaultdict(list)

    for ann in item["annotations"]:
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

    # Smooth bbox centers across target frames to reduce YOLO jitter
    smoothed_centers: dict[int, tuple[float, float]] = {}
    if crop_size:
        annotated = [
            (f, int(f * ls_fps / fps))
            for f in sorted(target_frames)
            if frame_annotations[int(f * ls_fps / fps)]
        ]
        if len(annotated) >= 2:
            k = min(5, len(annotated))
            kernel = np.ones(k) / k
            cx_seq = np.array(
                [float(frame_annotations[af][0].split()[1]) for _, af in annotated]
            )
            cy_seq = np.array(
                [float(frame_annotations[af][0].split()[2]) for _, af in annotated]
            )
            cx_smooth = np.convolve(cx_seq, kernel, mode="same")
            cy_smooth = np.convolve(cy_seq, kernel, mode="same")
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

        stem = f"{video_filename.replace('.mp4', '')}_frame_{frame_num:05d}"
        frame_file = path_to_output_frames / f"{stem}.jpg"

        # Need to inverse the fps correction to get the correct annotation frame
        frame_num_ann = int(frame_num * ls_fps / fps)
        lines = frame_annotations[frame_num_ann]

        if crop_size:
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
) -> None:
    """Stream a large JSON annotation file and convert it to YOLO format.

    This function streams a potentially large JSON annotation file using
    `ijson` and converts all video annotations into YOLO-format `.txt` files.
    Processing can be done either serially or in parallel using multiprocessing.

    Args:
        path_to_videos: Root directory containing all referenced video files.
        path_to_annotations: Path to the JSON annotation file to be processed.
        path_to_output: Output directory where YOLO `.txt` files will be written.
        processes: Number of worker processes to use. If set to 1, processing
            runs serially (useful for testing and debugging). Values greater
            than 1 enable multiprocessing.

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
                )
        else:
            # Parallel execution
            worker_func = partial(
                process_item,
                path_to_videos=path_to_videos,
                path_to_output=path_to_output,
                path_to_guidance=path_to_guidance,
                crop_size=crop_size,
            )

            with Pool(processes=processes) as pool:
                for output_file in tqdm(
                    pool.imap(worker_func, items, chunksize=1),
                    desc="Processing videos",
                ):
                    if output_file:
                        print(f"Processed {output_file}")
