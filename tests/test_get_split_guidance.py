import pytest
import polars as pl
from pathlib import Path
from bird_cv.get_split_guidance import (
    split_camera_data,
    subsample_frames,
    sample_resting_frames,
)
from typing import Dict


@pytest.fixture
def video_data_path(tmp_path: Path) -> Path:
    """Creates a small fake video NDJSON file for testing.

    Args:
        tmp_path (Path): pytest temporary directory.

    Returns:
        Path: Path to the generated video NDJSON file.
    """
    video_data = pl.DataFrame(
        {
            "video_id": ["v1", "v2"],
            "camera_id": ["c1", "c2"],
            "framesCount": [10, 12],
            "duration": [1.0, 1.2],
            "video_path": ["path1.mp4", "path2.mp4"],
        }
    )
    path = tmp_path / "video.ndjson"
    video_data.write_ndjson(path)
    return path


@pytest.fixture
def frame_data_path(tmp_path: Path) -> Path:
    """Creates a small fake frame NDJSON file for testing.

    Args:
        tmp_path (Path): pytest temporary directory.

    Returns:
        Path: Path to the generated frame NDJSON file.
    """
    frame_data = pl.DataFrame(
        {
            "video_id": ["v1", "v1", "v2"],
            "frame_begin": [1, 4, 1],
            "frame_end": [3, 5, 5],
            "label": ["A", "Resting", "B"],
            "framesCount": [10, 10, 12],
        }
    )
    path = tmp_path / "frames.ndjson"
    frame_data.write_ndjson(path)
    return path


def test_subsample_frames() -> None:
    """Test that subsample_frames returns the expected columns.

    Verifies that the output DataFrame contains 'video_id', 'target_frames',
    and 'label_counts' columns, and that only unique video_ids are preserved.
    """
    df = pl.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "frame_begin": [1, 5],
            "frame_end": [2, 5],
            "label": ["A", "Resting"],
            "framesCount": [10, 10],
        }
    )
    result = subsample_frames(df)
    assert "video_id" in result.columns
    assert "target_frames" in result.columns
    assert "label_counts" in result.columns
    assert result.shape[0] == 1  # only one unique video_id


def test_sample_resting_frames() -> None:
    """Test that sample_resting_frames correctly adds resting frames.

    Checks that the target_frames column is extended with additional frames
    based on the max count of other labels.
    """
    df = pl.DataFrame(
        {
            "video_id": ["v1"],
            "target_frames": [[1, 2]],
            "A": [2],
            "B": [0],
            "framesCount": [5],
        }
    )
    df = df.with_columns(label_counts=pl.struct("A", "B")).drop("A", "B")
    result = sample_resting_frames(df, seed=42)
    assert "target_frames" in result.columns
    # target_frames should now include additional frames
    assert len(result["target_frames"][0]) > 2


def test_split_camera_data(
    tmp_path: Path, video_data_path: Path, frame_data_path: Path
) -> None:
    """Test that split_camera_data creates parquet files for train/val splits.

    Args:
        tmp_path (Path): pytest temporary directory.
        video_data_path (Path): Path to fake video NDJSON file fixture.
        frame_data_path (Path): Path to fake frame NDJSON file fixture.

    Checks that parquet files are created for each split and contain expected columns.
    """
    output_path = tmp_path
    split_ratio: Dict[str, float] = {"train": 0.5, "val": 0.5, "test": 0.0}

    split_camera_data(
        video_data_path, frame_data_path, output_path, split_ratio, random_seed=0
    )

    train_file = output_path / "train_lookup.parquet"
    val_file = output_path / "val_lookup.parquet"

    # Parquet files should exist
    assert train_file.exists()
    assert val_file.exists()

    # Load one and check columns
    df = pl.read_parquet(train_file)
    assert "video_id" in df.columns
    assert "target_frames" in df.columns
    assert "fps" in df.columns
