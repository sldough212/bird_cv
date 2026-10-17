"""Per-video, per-stage latency tracking written to ``timings.parquet``."""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from bird_cv.core.schemas import TIMINGS_SCHEMA


def _gpu_name() -> str:
    """Best-effort GPU name: CUDA device name if torch sees one, else SLURM's GPU list."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return os.environ.get("SLURM_JOB_GPUS", "cpu")


class StageTimer:
    """Collects wall-clock timings for pipeline stages.

    Example::

        timer = StageTimer()
        with timer.stage("L25", "IMG_0016", "sam2", frames=900):
            run_sam2(...)
        timer.write(run_dir / "timings.parquet")
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.gpu = _gpu_name()

    @contextmanager
    def stage(
        self, camera_id: str, video_id: str, stage: str, frames: int
    ) -> Iterator[None]:
        """Time the enclosed block and record it as one row."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append(
                {
                    "camera_id": camera_id,
                    "video_id": video_id,
                    "stage": stage,
                    "seconds": time.perf_counter() - start,
                    "frames": frames,
                    "gpu": self.gpu,
                }
            )

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.rows, schema=TIMINGS_SCHEMA)

    def write(self, path: Path) -> None:
        """Write all recorded rows to ``path`` as parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().write_parquet(path)
