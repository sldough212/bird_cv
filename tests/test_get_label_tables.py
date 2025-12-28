"""Test for converstion of label studio tables to frame-level ndjson format."""

import polars as pl
from pathlib import Path

from bird_cv.get_label_tables import get_label_tables


def test_get_label_tables_creates_expected_outputs(tmp_path: Path) -> None:
    """Verify that label tables are correctly generated from a Label Studio export.

    This test creates a minimal, structurally valid Label Studio JSON export,
    executes `get_label_tables`, and asserts that:

    - The expected output NDJSON files are created
    - Video-level metadata is written correctly
    - Frame-level annotation data is written correctly
    - URL-encoded paths (e.g., `%2C`) are properly decoded

    Args:
        tmp_path (Path): Pytest-provided temporary directory for test isolation.

    Returns:
        None
    """
    # Arrange: minimal Label Studio-like JSON export
    label_json = tmp_path / "labels.json"
    label_json.write_text(
        """
        [
          {
            "inner_id": 1,
            "total_annotations": 1,
            "data": {
              "video": "/data/2021_bunting_clips/H7%2CI22/08.mp4",
              "framesCount": 100,
              "duration": 3.2
            },
            "annotations": [
              {
                "id": 10,
                "result": [
                  {
                    "id": "r1",
                    "value": {
                      "labels": ["bird"],
                      "sequence": [
                        {
                          "frame": 1,
                          "x": 10,
                          "y": 20,
                          "width": 30,
                          "height": 40
                        }
                      ]
                    }
                  }
                ]
              }
            ]
          }
        ]
        """
    )

    output_dir = tmp_path / "output"

    # Act
    get_label_tables(label_json_path=label_json, output_dir=output_dir)

    # Assert: output files exist
    video_path = output_dir / "video_data.ndjson"
    frame_path = output_dir / "frame_data.ndjson"

    # Assert: video data
    videos = pl.read_ndjson(video_path)
    assert videos.shape == (1, 4)
    assert videos["video_id"][0] == 1
    assert videos["video_path"][0] == "/H7,I22/08.mp4"
    assert videos["framesCount"][0] == 100
    assert videos["duration"][0] == 3.2

    # Assert: frame data
    frames = pl.read_ndjson(frame_path)
    assert frames.shape == (1, 8)
    assert frames["video_id"][0] == 1
    assert frames["track_id"][0] == "r1"
    assert frames["label"][0] == "bird"
    assert frames["frame"][0] == 1
