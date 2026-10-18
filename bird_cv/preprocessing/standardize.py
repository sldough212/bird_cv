"""Re-encode one raw video to the standardized 15 fps H.264 video.

The standardized video's frames define the canonical frame index (see
``docs/adr/0002-canonical-frame-index.md``).
"""

import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import imageio_ffmpeg

from bird_cv.core.frames import STANDARD_FPS
from bird_cv.core.ids import parse_raw_video_path


def standardize_video(
    video_path: Path,
    output_path: Path,
    fps: float = STANDARD_FPS,
    max_frames: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Re-encode a video to H.264/mp4 at ``fps``, optionally truncated.

    Downsamples via ffmpeg's ``fps`` filter (frame-accurate drop/duplicate for
    any source frame rate) and pads odd width/height up by 1px, since libx264
    requires even dimensions. Always re-encodes to H.264/yuv420p because some
    source AVIs fail to reopen with ffmpeg's default AVI codec.

    Args:
        video_path: Raw video to read.
        output_path: Destination ``.mp4``; parent directories are created.
        fps: Output frame rate. Defaults to :data:`STANDARD_FPS`.
        max_frames: If set, keep at most this many output frames.

    Returns:
        The finished ffmpeg process; callers decide how to handle a non-zero
        ``returncode`` (``stderr`` holds ffmpeg's log).
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)
    ffmpeg_cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2",
    ]
    if max_frames is not None:
        ffmpeg_cmd += ["-frames:v", str(max_frames)]
    ffmpeg_cmd += [
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    return subprocess.run(ffmpeg_cmd, capture_output=True)


def expected_standard_frames(
    header_frames: int,
    header_fps: float,
    fps: float = STANDARD_FPS,
    max_frames: int | None = None,
) -> int:
    """Frames :func:`standardize_video` writes for a video, from its header.

    ffmpeg's ``fps`` filter keeps one output frame per ``1/fps`` of input
    duration, rounding half up (``round=near``): ``floor(n * fps /
    header_fps + 0.5)``, then ``-frames:v`` caps it at ``max_frames``.
    Checked against ffmpeg on 12, 15, 24, 25, 29.97 and 30 fps sources.

    Args:
        header_frames: Frame count in the raw video's header.
        header_fps: Frame rate in the raw video's header.
        fps: Output frame rate, as passed to :func:`standardize_video`.
        max_frames: Frame cap, as passed to :func:`standardize_video`.

    Returns:
        The standardized video's frame count.

    Raises:
        ValueError: If ``header_fps`` isn't positive.
    """
    if header_fps <= 0:
        raise ValueError(f"header fps must be positive, got {header_fps}")
    n = math.floor(header_frames * fps / header_fps + 0.5)
    return n if max_frames is None else min(n, max_frames)


def video_header(video_path: Path) -> tuple[float, int]:
    """``(fps, frame count)`` from a video's header, via OpenCV (0s if unreadable)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, count


def find_raw_videos(
    raw_videos_path: Path,
    cameras: Sequence[str] | None = None,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    """Raw videos under ``{raw_videos_path}/{camera_id}/``, sorted, optionally filtered.

    Args:
        raw_videos_path: Root with one folder per camera.
        cameras: Only these camera ids, if given.
        extensions: Lower-case file suffixes counted as videos (the same set
            ``check_video_specs`` scans).

    Returns:
        Matching video paths, sorted.
    """
    videos = sorted(
        p
        for p in raw_videos_path.glob("*/*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if cameras is not None:
        videos = [v for v in videos if parse_raw_video_path(v)[0] in cameras]
    return videos
