"""Image cropping, label normalization, and video-encoding utilities for YOLO preprocessing."""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import polars as pl
from PIL import Image

from bird_cv.utils import extract_camera_video


def crop_and_mask_image(img, mask, black_out=True, padding=0):
    """
    Crop an image to the bounding box of the True region in a mask.

    Args:
        img (PIL.Image.Image): Input image (RGB or grayscale)
        mask (np.ndarray): Boolean array same size as image (True = object)
        black_out (bool): Whether to zero out pixels outside mask
        padding (int): Optional pixels to pad around the bounding box

    Returns:
        cropped_img (PIL.Image.Image): Cropped (and masked) image
        crop_coords (tuple): (x_min, y_min, x_max, y_max) in original image coords
    """
    img_np = np.array(img)
    mask = np.array(mask, dtype=bool)

    ys, xs = np.where(mask)
    y_min, y_max = (
        max(ys.min() - padding, 0),
        min(ys.max() + padding, mask.shape[0] - 1),
    )
    x_min, x_max = (
        max(xs.min() - padding, 0),
        min(xs.max() + padding, mask.shape[1] - 1),
    )

    if black_out:
        masked_img = img_np.copy()
        masked_img[~mask] = 0
    else:
        masked_img = img_np

    cropped = masked_img[y_min : y_max + 1, x_min : x_max + 1]
    cropped_img = Image.fromarray(cropped)

    return cropped_img, (x_min, y_min, x_max, y_max)


def normalize_labels_for_crop(
    labels, crop_coords, image_shape, return_winning_idx: bool = False
):
    """
    Convert bird bounding boxes to YOLO format relative to a cropped image.

    Args:
        labels (list): List of [category, [x_center, y_center, width, height]] in normalized coords (0-1)
        crop_coords (tuple): (x_min, y_min, x_max, y_max) of crop in original pixels
        image_shape (tuple): (height, width) of original image in pixels
        return_winning_idx (bool): If True, also return the index of the last label
            whose centroid falls within the crop.

    Returns:
        normalized_labels (list): List of [category, x_center, y_center, width, height] normalized 0-1
    """
    y_min, x_min = crop_coords[1], crop_coords[0]
    y_max, x_max = crop_coords[3], crop_coords[2]
    img_h, img_w = image_shape

    cropped_w = x_max - x_min
    cropped_h = y_max - y_min

    normalized_labels = []
    winning_idx: int | None = None
    for ii, label in enumerate(labels):
        category, bbox = label
        # bbox assumed to be normalized to original image: [x_center, y_center, w, h]
        x0, y0, w, h = bbox

        # Convert to pixel coords in full image
        cx_pixel = x0 * img_w
        cy_pixel = y0 * img_h
        w_pixel = w * img_w
        h_pixel = h * img_h

        # Offset relative to cropped image
        cx_crop = cx_pixel - x_min
        cy_crop = cy_pixel - y_min

        # Normalize relative to cropped image
        cx_norm = cx_crop / cropped_w
        cy_norm = cy_crop / cropped_h
        w_norm = w_pixel / cropped_w
        h_norm = h_pixel / cropped_h

        # Only include labels that are at least partially inside crop
        if 0 <= cx_norm <= 1 and 0 <= cy_norm <= 1:
            normalized_labels.append([category, cx_norm, cy_norm, w_norm, h_norm])
            winning_idx = ii

    if return_winning_idx:
        return normalized_labels, winning_idx
    return normalized_labels


def images_to_video(image_dir: Path, output_path: Path, fps: float = 30) -> None:
    """Encode a directory of JPEG frames into an H.264 MP4 using ffmpeg.

    Reads all ``*.jpg`` files from ``image_dir`` in sorted order and pipes
    them as raw BGR frames to ffmpeg. Frames are padded to even dimensions
    if necessary for H.264 compatibility.

    Args:
        image_dir: Directory containing sequentially named JPEG frames.
        output_path: Destination path for the output MP4 file.
        fps: Frame rate of the output video.

    Raises:
        FileNotFoundError: If no JPEG files are found in ``image_dir``.
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    frames = sorted(image_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No JPGs found in {image_dir}")

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    # Round dimensions up to the nearest even number
    new_w = w + (w % 2)
    new_h = h + (h % 2)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{new_w}x{new_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        # Pad bottom/right by 1px if needed, to match new_w/new_h
        if frame.shape[1] != new_w or frame.shape[0] != new_h:
            frame = cv2.copyMakeBorder(
                frame,
                top=0,
                bottom=new_h - frame.shape[0],
                left=0,
                right=new_w - frame.shape[1],
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")


def run_images_to_video(
    split_guidance_path: Path,
    clip_output_path: Path,
    video_output_path: Path,
) -> None:
    """Convert cropped cage JPEG sequences into MP4 videos for every cage and video.

    Reads the split guidance to determine FPS per video, then iterates over
    all cage directories under ``clip_output_path/{camera_id}/{video_id}/``
    and encodes each into an MP4 at ``video_output_path/{camera_id}/{video_id}/{cage_id}.mp4``.

    Args:
        split_guidance_path: Path to the split guidance parquet supplying
            ``video_path`` and ``fps`` columns.
        clip_output_path: Root directory of cropped cage JPEG images produced
            by ``crop_cages``.
        video_output_path: Root directory where output MP4 files will be written.
    """
    split_guidance = pl.read_parquet(split_guidance_path)

    for video_str, fps in split_guidance.select("video_path", "fps").iter_rows():
        camera_id, video_name = extract_camera_video(video_str=video_str)
        video_id = Path(video_name).stem
        image_output_path = clip_output_path / camera_id / video_id

        if not image_output_path.exists():
            continue

        # Iterate through the cages
        for cage_path in image_output_path.iterdir():
            if not cage_path.is_dir():
                continue

            if not any(cage_path.iterdir()):
                continue

            cage_output_path = (
                video_output_path / camera_id / video_id / f"{cage_path.name}.mp4"
            )
            cage_output_path.parent.mkdir(exist_ok=True, parents=True)

            images_to_video(image_dir=cage_path, fps=fps, output_path=cage_output_path)


def check_video_specs(
    videos_path: Path,
    output_path: Path,
    max_time: float = 60,
    expected_fps: float = 15,
    max_frames: int = 900,
) -> None:
    """Flag out-of-spec videos and write standardized copies for every video found.

    Recursively scans ``videos_path`` for video files and, for each one, reads
    its duration (via frame count / FPS) and FPS via OpenCV, printing the path of
    any video that exceeds ``max_time`` seconds or whose FPS does not match
    ``expected_fps``. Regardless of the original spec, every video is then
    re-encoded via ffmpeg to H.264/mp4 at ``expected_fps`` and truncated to at
    most ``max_frames`` frames, and written to ``output_path`` mirroring the
    directory structure found under ``videos_path`` (with each output file's
    suffix normalized to ``.mp4``, regardless of the source container).

    Args:
        videos_path: Root directory to search recursively for video files.
        output_path: Root directory to mirror ``videos_path`` into with
            standardized (downsampled/truncated) copies of each video.
        max_time: Maximum allowed video duration in seconds, used for the
            spec-check print only.
        expected_fps: Required frame rate in frames per second. Also used as
            the target FPS for the standardized output.
        max_frames: Maximum number of frames to keep in each standardized
            output video (defaults to 900, i.e. 60s at 15fps).
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    for p in videos_path.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in video_extensions):
            continue

        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if not fps:
            print(f"{p}: could not determine FPS")
        else:
            duration = frame_count / fps
            if duration > max_time or fps != expected_fps:
                print(f"{p}: duration={duration:.2f}s, fps={fps:.2f}")

        out_path = (output_path / p.relative_to(videos_path)).with_suffix(".mp4")
        out_path.parent.mkdir(exist_ok=True, parents=True)

        # Downsample to expected_fps (via ffmpeg's fps filter, which handles
        # arbitrary source frame rates by frame-accurate drop/duplicate), cap
        # the output at max_frames, and re-encode explicitly to H.264/mp4.
        # Some source AVIs fail to reopen if left to ffmpeg's default codec
        # for the AVI muxer, so always normalize to a known-good format
        # (mirroring the codec choice in images_to_video above). libx264
        # requires even dimensions, so pad odd width/height up by 1px.
        ffmpeg_cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(p),
            "-vf",
            f"fps={expected_fps},pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-frames:v",
            str(max_frames),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(out_path),
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True)
        if result.returncode != 0:
            print(f"{p}: ffmpeg failed - {result.stderr.decode(errors='replace')}")
