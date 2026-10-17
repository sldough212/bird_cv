"""Re-encode one raw video to the standardized 15 fps H.264 video.

The standardized video's frames define the canonical frame index (see
``docs/adr/0002-canonical-frame-index.md``).
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg

from bird_cv.core.frames import STANDARD_FPS


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
