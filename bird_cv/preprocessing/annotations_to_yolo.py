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

from bird_cv.core.frames import FrameMap, frame_map_path, is_aligned_with_standard


def crop_bird(
    frame: np.ndarray, cx_norm: float, cy_norm: float, crop_size: int = 224
) -> np.ndarray:
    """Crop a fixed square centered on a bird, padding with black past the edges.

    Args:
        frame: Frame as an ``(H, W[, C])`` array (e.g. BGR from cv2).
        cx_norm: Crop center x, normalized 0-1 to the frame width.
        cy_norm: Crop center y, normalized 0-1 to the frame height.
        crop_size: Side length in pixels of the square crop.

    Returns:
        The ``crop_size`` x ``crop_size`` crop (for even ``crop_size``).
    """
    frame_h, frame_w = frame.shape[:2]
    half = crop_size // 2

    cx = int(cx_norm * frame_w)
    cy = int(cy_norm * frame_h)

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
    return crop


def smooth_centers(
    cx: np.ndarray, cy: np.ndarray, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth box-center sequences with an edge-padded moving average.

    The window shrinks to ``min(window, len(cx))`` for short sequences.

    Args:
        cx: Box x-centers, one per frame, in frame order.
        cy: Box y-centers, one per frame, the same length as ``cx``.
        window: Moving-average window length in frames; at least 1.

    Returns:
        Smoothed ``(cx, cy)``, each the same length as the input (empty
        arrays for empty input).

    Raises:
        ValueError: If ``window`` is less than 1.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    n = len(cx)
    if n == 0:
        return np.empty(0), np.empty(0)
    k = min(window, n)
    kernel = np.ones(k) / k
    pad = k // 2
    cx_smooth = np.convolve(np.pad(cx, pad, mode="edge"), kernel, mode="valid")
    cy_smooth = np.convolve(np.pad(cy, pad, mode="edge"), kernel, mode="valid")
    # An even window yields one extra trailing value; drop it
    return cx_smooth[:n], cy_smooth[:n]


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
    frame_map_root: Path | None = None,
    by_cage: bool = False,
) -> bool:
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
        path_to_output: Directory where output files will be written. In YOLO
            mode, one image and label file is created per frame. In crop mode,
            one cropped JPEG is saved per annotated frame.
        path_to_guidance: Path to the split guidance parquet used to filter
            videos and determine train/val/test split assignment.
        crop_size: If provided, saves a square crop of this side length (in
            pixels) centered on each bird bbox instead of the full frame and
            YOLO label. Bbox centers are smoothed across frames to reduce
            YOLO jitter before cropping.
        frame_map_root: Crop mode only. If provided, also write a
            :class:`~bird_cv.core.frames.FrameMap` giving the canonical frame
            of each saved crop frame, at
            ``frame_map_path(frame_map_root / camera_id / video_id / f"{cage}.mp4")``:
            next to the bird-crop video when ``frame_map_root`` is the
            ``video_output_path`` later given to ``run_images_to_video``.
            Assumes the cage video keeps every canonical frame
            (``crop_cages(keep_empty_frames=True)``). Not written if a frame
            has more than one box, since the crops then don't form one
            frame per canonical frame, nor if no frame has a box. Also
            skipped if the cage video or the guidance fps isn't aligned with
            the standardized video (``is_aligned_with_standard``).
        by_cage: Match guidance rows on a ``cage_id`` column too, so each
            cage video gets its own ``target_frames`` (default: one row per
            camera video, shared by its cages).

    Returns:
        True if a frame map was skipped because the video isn't aligned,
        else False. Files are written to disk as a side effect.

    Raises:
        ValueError: If ``frame_map_root`` is given without ``crop_size``.
    """
    if frame_map_root is not None and not crop_size:
        raise ValueError("frame_map_root requires crop_size")
    # Get video path and clean it
    video_path = (
        item["data"]["video"].split(f"{path_to_videos.name}/")[-1].replace("%2C", ",")
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
    if by_cage:
        video_guidance = video_guidance.filter(pl.col("cage_id") == cage)

    print(f"Working on {video_path}")
    if video_guidance.is_empty():
        print(f"Video {video_path} not in target")
        return False

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

    # Canonical frame of each saved crop frame, in order (for the frame map)
    crop_canonical: list[int] = []
    multi_box = False

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
        frame_num_ann = int(frame_num * ls_fps / fps)
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
            if lines:
                # frame_num counts video frames from 1
                crop_canonical.append(frame_num - 1)
                multi_box = multi_box or len(lines) > 1
        else:
            stem = f"{camera_id}_{video_id}_{cage}_{frame_num:05d}"
            frame_file = path_to_output_frames / f"{stem}.jpg"
            cv2.imwrite(str(frame_file), frame)
            label_file = path_to_output_labels / f"{stem}.txt"
            with open(label_file, "w") as f:
                f.write("\n".join(lines))

        frame_num += 1

    cap.release()

    if crop_size and frame_map_root is not None and crop_canonical:
        # frame_num ends one past the number of frames read
        n_read = frame_num - 1
        aligned = (
            fps > 0
            and ls_fps > 0
            and is_aligned_with_standard(n_read, n_read / fps)
            and is_aligned_with_standard(n_read, n_read / ls_fps)
        )
        if not aligned:
            print(f"Not writing frame map for {video_path}: not aligned")
            return True
        if multi_box:
            print(f"Not writing frame map for {video_path}: frames with >1 box")
        else:
            FrameMap(tuple(crop_canonical)).write(
                frame_map_path(frame_map_root / camera_id / video_id / f"{cage}.mp4")
            )
    return False


def stream_annotations_to_yolo(
    path_to_videos: Path,
    path_to_annotations: Path,
    path_to_output: Path,
    path_to_guidance: Path,
    processes: int = 4,
    crop_size: int | None = None,
    *,
    frame_map_root: Path | None = None,
    by_cage: bool = False,
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
        crop_size: If provided, save bird crops instead of YOLO frames and
            labels (see :func:`process_item`).
        frame_map_root: Crop mode only: root under which to write a frame map
            per bird-crop video (see :func:`process_item`).
        by_cage: Guidance has one row per cage video (see :func:`process_item`).

    Raises:
        ValueError: If ``frame_map_root`` is given without ``crop_size``.

    Returns:
        None. YOLO annotation files are written to disk as a side effect.
    """
    if frame_map_root is not None and not crop_size:
        raise ValueError("frame_map_root requires crop_size")

    # Create direcotries for training
    path_to_output.mkdir(exist_ok=True, parents=True)
    splits = ["train", "val", "test"]
    for split in splits:
        (path_to_output / "images" / split).mkdir(exist_ok=True, parents=True)
        (path_to_output / "labels" / split).mkdir(exist_ok=True, parents=True)

    n_unaligned = 0
    with open(path_to_annotations, "rb") as f:
        items = ijson.items(f, "item")  # generator over top-level items

        if processes == 1:
            # Serial execution (recommended for unit tests)
            for item in items:
                n_unaligned += process_item(
                    item=item,
                    path_to_videos=path_to_videos,
                    path_to_output=path_to_output,
                    path_to_guidance=path_to_guidance,
                    crop_size=crop_size,
                    frame_map_root=frame_map_root,
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
                frame_map_root=frame_map_root,
                by_cage=by_cage,
            )

            with Pool(processes=processes) as pool:
                for skipped in tqdm(
                    pool.imap(worker_func, items, chunksize=1),
                    desc="Processing videos",
                ):
                    n_unaligned += skipped

    if frame_map_root is not None:
        print(f"Skipped frame maps for {n_unaligned} unaligned videos")
