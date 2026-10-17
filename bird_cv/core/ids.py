"""Building ``camera_id`` / ``video_id`` / ``cage_id`` the same way everywhere.

Ground truth (parsed from Label Studio media paths) and predictions (built
from raw video paths and SAM2 config positions) are joined on these ids, so
both sides must produce identical strings. See
:data:`bird_cv.core.schemas.KEY_COLUMNS` for the formats.
"""

from pathlib import Path
from urllib.parse import unquote


def parse_media_path(path: str) -> tuple[str, str, str]:
    """Parse a cage or bird-crop video path into ``(camera_id, video_id, cage_id)``.

    Accepts Label Studio local-file URLs and plain paths whose last three
    parts are ``{camera}/{video}/{cage}.mp4``, e.g.
    ``/data/local-files/?d=gscratch/.../cage_videos/I35%2CL15/IMG_0016/0.mp4``.

    Args:
        path: Media path or Label Studio ``data.video`` URL.

    Returns:
        ``(camera_id, video_id, cage_id)``, e.g. ``("I35,L15", "IMG_0016", "0")``.
    """
    parts = Path(unquote(path.split("?d=")[-1])).parts
    camera, video, cage_file = parts[-3], parts[-2], parts[-1]
    return camera, video, Path(cage_file).stem


def parse_raw_video_path(path: str | Path) -> tuple[str, str]:
    """Parse a raw video path ``{camera}/{video}.{ext}`` into ``(camera_id, video_id)``.

    Args:
        path: Path to a raw or standardized video.

    Returns:
        ``(camera_id, video_id)``, e.g. ``("H7,I22", "08")``.
    """
    p = Path(unquote(str(path)))
    return p.parent.name, p.stem


def cage_id_from_index(index: int) -> str:
    """Cage id for the cage at position ``index`` in a camera's SAM2 config."""
    return str(index)
