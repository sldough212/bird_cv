from pathlib import Path

import polars as pl

from bird_cv.core.schemas import TIMINGS_SCHEMA
from bird_cv.evaluation.plots import (
    plot_confusion,
    plot_per_class,
    plot_timeline,
    plot_timings,
)


def test_plots_write_pngs(tmp_path: Path) -> None:
    joined = pl.DataFrame(
        {
            "camera_id": ["C"] * 6,
            "video_id": ["v"] * 6,
            "cage_id": ["0", "0", "0", "1", "1", "1"],
            "frame": [0, 1, 2, 0, 1, 2],
            "label": ["resting", "resting", None, "no_bird", "preening", "preening"],
            "pred_label": [
                "resting",
                "head_move",
                "resting",
                None,
                "preening",
                "no_bird",
            ],
        }
    )
    timings = pl.DataFrame(
        [("C", "v", "sam2", 2.0, 900, "L40S"), ("C", "v", "detect", 1.0, 900, "L40S")],
        schema=TIMINGS_SCHEMA,
        orient="row",
    )
    plot_timeline(joined, tmp_path / "t.png")
    plot_confusion(joined, tmp_path / "c.png")
    plot_per_class(joined, tmp_path / "p.png")
    plot_timings(timings, tmp_path / "s.png")
    assert all(
        (tmp_path / f).stat().st_size > 0 for f in ("t.png", "c.png", "p.png", "s.png")
    )
