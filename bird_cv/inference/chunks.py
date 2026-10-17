"""Fixed-size chunking of a standardized video and stitching of overlapping chunks.

A video is processed ``chunk_frames`` canonical frames at a time so memory
stays bounded on long videos. For the classifier, each chunk's window also
reaches ``overlap`` frames back into the previous chunk, so behaviors that
span a chunk boundary keep temporal context; :func:`stitch_overlapping`
then gives every canonical frame exactly one prediction.
"""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np

from bird_cv.segmentation.frames import iter_canonical_frames

T = TypeVar("T")


def iter_chunks(
    video_path: Path, chunk_frames: int, overlap: int = 0
) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Read a standardized video in chunks of canonical frames.

    Chunk ``k`` owns canonical frames ``[k * chunk_frames, (k + 1) * chunk_frames)``
    and is yielded together with up to ``overlap`` frames preceding it.

    Args:
        video_path: Standardized video.
        chunk_frames: Frames per chunk.
        overlap: Frames of the previous chunk to prepend to each chunk.

    Yields:
        ``(window_start, frames)``: the canonical frame of ``frames[0]`` and
        the BGR frames of the window, in order.
    """
    if not 0 <= overlap < chunk_frames:
        raise ValueError("overlap must be in [0, chunk_frames)")
    tail: list[np.ndarray] = []
    chunk: list[np.ndarray] = []
    chunk_start = 0
    for frame_idx, frame in iter_canonical_frames(video_path):
        if not chunk:
            chunk_start = frame_idx
        chunk.append(frame)
        if len(chunk) == chunk_frames:
            yield chunk_start - len(tail), tail + chunk
            tail = chunk[len(chunk) - overlap :] if overlap else []
            chunk = []
    if chunk:
        yield chunk_start - len(tail), tail + chunk


def stitch_overlapping(
    windows: Sequence[tuple[int, int, Mapping[int, T]]],
) -> dict[int, T]:
    """Merge per-window predictions keyed by canonical frame into one per frame.

    Where consecutive windows overlap, the overlap is split at its midpoint:
    the earlier window keeps the frames before it and the later window the
    rest, so each frame's prediction comes from the window in which it sits
    furthest from an edge.

    Args:
        windows: ``(start, end, predictions)`` per window, sorted by
            ``start``; ``end`` is exclusive and ``predictions`` maps canonical
            frames in ``[start, end)`` to a prediction (frames may be absent).

    Returns:
        Mapping from canonical frame to the prediction of the window that
        owns it.
    """
    stitched: dict[int, T] = {}
    for k, (start, end, predictions) in enumerate(windows):
        own_start = start if k == 0 else (start + windows[k - 1][1]) // 2
        own_end = end if k == len(windows) - 1 else (windows[k + 1][0] + end) // 2
        stitched.update(
            {f: p for f, p in predictions.items() if own_start <= f < own_end}
        )
    return stitched
