"""Check that 224 px recrops of the cage videos match the Classify bird-crop videos.

Recrops each sampled cage's boxed frames from its Detect cage video, as
build_datasets does for bird crops other than 224 px (ADR 0010), and compares
them, unresized, with the Classify bird-crop video of that cage. Exits 1 if a
cage's mean absolute pixel difference exceeds ``--max-mean-diff``.

Usage::

    python bird_cv/pipelines/check_recrop.py CONFIG [--dataset ID] [--cages N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from bird_cv.core.ids import get_local_path, parse_media_path
from bird_cv.core.layout import Layout, snapshot_files
from bird_cv.pipelines.config import load_config
from bird_cv.preprocessing.annotations_to_mae import extract_video_frames
from bird_cv.preprocessing.crop import crop_bird, ground_truth_crop_centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--dataset", default=None, help="Dataset id")
    parser.add_argument("--cages", type=int, default=10)
    parser.add_argument("--max-mean-diff", type=float, default=8.0)
    args = parser.parse_args()

    cfg = load_config(args.config_path, require_api_key=False)
    layout = Layout(cfg.path_to_base_output)
    dataset_id = layout.resolve_dataset(args.dataset, None)
    detect, classify = (
        {
            parse_media_path(t["data"]["video"]): get_local_path(t["data"]["video"])
            for t in json.loads(path.read_text())
        }
        for path in snapshot_files(layout.annotations(dataset_id))
    )
    centers = ground_truth_crop_centers(
        pl.read_parquet(layout.ground_truth(dataset_id)), set(detect) & set(classify)
    )
    keys = sorted(centers)[: args.cages]
    failed = 0
    for key in keys:
        cage = extract_video_frames(detect[key])
        crops = extract_video_frames(classify[key])
        frames = sorted(centers[key])
        n = min(len(frames), len(crops))
        diffs = [
            np.abs(
                crop_bird(cage[f], *centers[key][f], crop_size=224).astype(int)
                - crops[i].astype(int)
            ).mean()
            for i, f in enumerate(frames[:n])
            if f < len(cage) and crops[i].shape[:2] == (224, 224)
        ]
        mean = float(np.mean(diffs)) if diffs else float("nan")
        ok = bool(diffs) and mean <= args.max_mean_diff
        failed += not ok
        print(
            f"{'/'.join(key)}: {len(frames)} boxed, {len(crops)} crop frames,"
            f" {len(diffs)} compared, mean |diff| {mean:.2f},"
            f" max frame {max(diffs, default=float('nan')):.2f}"
            f" {'ok' if ok else 'MISMATCH'}"
        )
    print(f"{len(keys) - failed} of {len(keys)} cages match")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
