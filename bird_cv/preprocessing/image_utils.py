"""Image cropping, label normalization, and video-encoding utilities for YOLO preprocessing."""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import polars as pl
from PIL import Image

from bird_cv.core.ids import parse_raw_video_path


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

    # Compute max dimensions across all frames, not just the first,
    # so every frame only ever needs padding (never resizing/distortion)
    max_h, max_w = 0, 0
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")
        fh, fw = frame.shape[:2]
        max_h, max_w = max(max_h, fh), max(max_w, fw)

    # Round dimensions up to the nearest even number
    new_w = max_w + (max_w % 2)
    new_h = max_h + (max_h % 2)

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
        if frame is None:
            raise ValueError(f"Failed to read frame: {frame_path}")

        fh, fw = frame.shape[:2]
        if fh != new_h or fw != new_w:
            frame = cv2.copyMakeBorder(
                frame,
                top=0,
                bottom=new_h - fh,
                left=0,
                right=new_w - fw,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()


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
        camera_id, video_id = parse_raw_video_path(video_str)
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
