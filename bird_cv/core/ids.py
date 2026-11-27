"""Building ``camera_id`` / ``video_id`` / ``cage_id`` the same way everywhere.

Ground truth (parsed from Label Studio media paths) and predictions (built
from raw video paths and SAM2 config positions) are joined on these ids, so
both sides must produce identical strings. See
:data:`bird_cv.core.schemas.KEY_COLUMNS` for the formats.
"""

from pathlib import Path
from urllib.parse import unquote

import polars as pl


def get_local_path(video_url: str) -> Path:
    """Absolute path of a Label Studio ``/data/local-files/?d=<path>`` URL.

    Plain paths (no ``?d=``) are URL-decoded and made absolute the same way.
    """
    return Path("/" + unquote(video_url.split("?d=")[-1]).lstrip("/"))


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
    parts = get_local_path(path).parts
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


def with_raw_video_ids(df: pl.DataFrame, column: str = "video_path") -> pl.DataFrame:
    """``df`` with ``camera_id`` and ``video_id`` parsed from raw video paths.

    Args:
        df: Table whose ``column`` holds ``{camera}/{video}.{ext}`` paths, as
            in split guidance.
        column: The path column.

    Returns:
        ``df`` plus ``camera_id`` and ``video_id`` from
        :func:`parse_raw_video_path`.
    """
    ids = [parse_raw_video_path(p) for p in df[column]]
    return df.with_columns(
        camera_id=pl.Series([c for c, _ in ids], dtype=pl.String),
        video_id=pl.Series([v for _, v in ids], dtype=pl.String),
    )
