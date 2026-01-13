import ijson
import cv2
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from typing import Any, Dict, Optional
import polars as pl
import numpy as np


def process_item(
    item: Dict[str, Any],
    path_to_videos: Path,
    path_to_output: Path,
    path_to_guidance: Path,
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

    # Load in the guidance
    guidance = pl.read_parquet(path_to_guidance)
    video_guidance = guidance.filter(
        pl.col("video_path")
        .str.split("2021_bunting_clips/")
        .list[-1]
        .str.replace_all("%2C", ",")
        == str(video_path)
    )

    print(f"Working on {video_path}")
    if video_guidance.is_empty():
        print(f"Video {video_path} not in target")
        return

    # Determine the split of the video based on guidance and redefine output paths
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
                frame_num: int = seq["frame"]

                # These are given as percentages of the entire frame
                x: float = seq["x"]
                y: float = seq["y"]
                w: float = seq["width"]
                h: float = seq["height"]

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

    target_frames = video_guidance.select("target_frames").to_numpy().squeeze()
    ls_fps = video_guidance.select("fps").item()

    # Open video to extract frames
    cap = cv2.VideoCapture(full_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Readjust the target frames back to the original video's FPS
    target_frames = np.unique(target_frames * fps / ls_fps).astype(int).tolist()

    # Extract that image frame and save
    frame_num = 1
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_num not in target_frames:
            frame_num += 1
            continue

        frame_file = (
            path_to_output_frames
            / f"{video_filename.replace('.mp4', '')}_frame_{frame_num:04d}.png"
        )
        cv2.imwrite(str(frame_file), frame)

        lines = frame_annotations[frame_num]
        label_file = (
            path_to_output_labels
            / f"{video_filename.replace('.mp4', '')}_frame_{frame_num:04d}.txt"
        )
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
                )
        else:
            # Parallel execution
            worker_func = partial(
                process_item,
                path_to_videos=path_to_videos,
                path_to_output=path_to_output,
                path_to_guidance=path_to_guidance,
            )

            with Pool(processes=processes) as pool:
                for output_file in tqdm(
                    pool.imap(worker_func, items, chunksize=1),
                    desc="Processing videos",
                ):
                    if output_file:
                        print(f"Processed {output_file}")
