"""Canonical frame conventions.

See ``docs/adr/0002-canonical-frame-index.md``. A canonical frame is the
0-indexed frame of the standardized video (``standard_fps`` in config,
15). Label Studio numbers
frames from 1; convert only at the Label Studio boundary, with
:func:`ls_to_canonical` and :func:`canonical_to_ls`.
"""


def ls_to_canonical(ls_frame: int) -> int:
    """Convert a 1-indexed Label Studio frame to a canonical frame.

    Only valid for media whose frames are one-to-one with the standardized
    video: 15 fps (Label Studio ``framesCount / duration``) with nothing
    dropped (``framesCount`` equals the standardized video's frame count).
    Check :func:`is_aligned_with_standard` before converting existing cage
    videos, which older ``crop_cages`` runs may have built with drops.
    """
    return ls_frame - 1


def is_aligned_with_standard(
    frames_count: int,
    duration: float,
    standard_fps: float,
    standard_frame_count: int | None = None,
    tolerance: float = 0.01,
) -> bool:
    """Whether Label Studio media lines up frame-for-frame with its standardized video.

    Args:
        frames_count: Label Studio ``framesCount`` for the media.
        duration: Label Studio ``duration`` in seconds.
        standard_fps: Frame rate of the standardized videos
            (``standard_fps`` in config).
        standard_frame_count: Frame count of the standardized source video,
            if known. When given, it must equal ``frames_count``.
        tolerance: Allowed relative fps deviation from ``standard_fps``.

    Returns:
        True if the fps matches and (when checkable) no frames were dropped.
    """
    if duration <= 0:
        return False
    fps_ok = abs(frames_count / duration - standard_fps) <= tolerance * standard_fps
    count_ok = standard_frame_count is None or standard_frame_count == frames_count
    return fps_ok and count_ok


def canonical_to_ls(frame: int) -> int:
    """Convert a canonical frame to a 1-indexed Label Studio frame."""
    return frame + 1
