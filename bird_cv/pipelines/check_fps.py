"""Report raw-header and Label Studio frame rates and counts of every Detect task.

Usage::

    python bird_cv/pipelines/check_fps.py CONFIG [--prepared ID | --detect-export PATH]
"""

import argparse
import json
from pathlib import Path

import polars as pl

from bird_cv.core.layout import Layout, snapshot_files
from bird_cv.pipelines.config import load_config
from bird_cv.preprocessing.check_fps import frame_rate_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--detect-export", type=Path, default=None)
    parser.add_argument("--prepared", default=None, help="Prepared-data id")
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    export = args.detect_export
    if export is None:
        layout = Layout(cfg.path_to_base_output)
        prepared = layout.resolve_prepared(args.prepared)
        export = snapshot_files(layout.snapshot(prepared.snapshot_hash))[0]
    table = frame_rate_table(
        cfg.path_to_raw_videos,
        json.loads(export.read_text()),
        max_frames=cfg.prepare.standardized_max_frames,
    )
    misaligned = table.filter(pl.col("aligned").not_())
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(table)
        print(
            f"{table.height} Detect tasks; {misaligned.height} misaligned "
            "(excluded from ground truth):"
        )
        print(misaligned)


if __name__ == "__main__":
    main()
