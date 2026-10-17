"""Canonical frame conventions and frame maps.

See ``docs/adr/0002-canonical-frame-index.md``. A canonical frame is the
0-indexed frame of the standardized (15 fps) video. Label Studio numbers
frames from 1; convert only at the Label Studio boundary, with
:func:`ls_to_canonical` and :func:`canonical_to_ls`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

STANDARD_FPS = 15
"""Frame rate every video is standardized to before processing."""


def ls_to_canonical(ls_frame: int) -> int:
    """Convert a 1-indexed Label Studio frame to a canonical frame."""
    return ls_frame - 1


def canonical_to_ls(frame: int) -> int:
    """Convert a canonical frame to a 1-indexed Label Studio frame."""
    return frame + 1


@dataclass(frozen=True)
class FrameMap:
    """Maps each frame of a derived video to its canonical frame.

    Position ``i`` of ``canonical`` is the canonical frame shown in frame
    ``i`` (0-indexed) of the derived video. For example, a bird-crop video
    built only from boxed frames 45..190 and 219..313 of its cage video has
    ``canonical == (45, 46, ..., 190, 219, ..., 313)``.

    Attributes:
        canonical: Canonical frame for each derived-video frame, strictly
            increasing.
    """

    canonical: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(b <= a for a, b in zip(self.canonical, self.canonical[1:])):
            raise ValueError("FrameMap.canonical must be strictly increasing")

    def __len__(self) -> int:
        return len(self.canonical)

    def to_canonical(self, derived_frame: int) -> int:
        """Return the canonical frame for a 0-indexed derived-video frame."""
        return self.canonical[derived_frame]

    @classmethod
    def identity(cls, n_frames: int) -> "FrameMap":
        """Frame map for a derived video that keeps every canonical frame."""
        return cls(tuple(range(n_frames)))

    def write(self, path: Path) -> None:
        """Write the map as JSON (conventionally ``<video>.frames.json``)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"canonical": list(self.canonical)}))

    @classmethod
    def read(cls, path: Path) -> "FrameMap":
        """Read a map written by :meth:`write`."""
        return cls(tuple(json.loads(path.read_text())["canonical"]))


def frame_map_path(video_path: Path) -> Path:
    """Conventional location of the frame map stored next to a video."""
    return video_path.with_suffix(".frames.json")
