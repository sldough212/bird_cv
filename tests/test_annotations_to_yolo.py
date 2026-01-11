import pytest
import json
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any, Tuple, Optional
import numpy as np
import polars as pl

from bird_cv.annotations_to_yolo import process_item, stream_annotations_to_yolo


@pytest.fixture
def sample_item() -> Dict[str, Any]:
    """Creates a small sample annotation item for testing.

    Returns:
        Dict: A single video annotation item with two frames annotated.
    """
    return {
        "data": {"video": "2021_bunting_clips/test_video.mp4"},
        "annotations": [
            {
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
                                    "height": 40,
                                },
                                {
                                    "frame": 2,
                                    "x": 50,
                                    "y": 60,
                                    "width": 70,
                                    "height": 80,
                                },
                            ],
                        },
                    }
                ]
            }
        ],
    }


@pytest.fixture
def path_to_guidance(tmp_path: Path) -> Path:
    """Creates a small fake guidance Parquet file for testing.

    Args:
        tmp_path (Path): pytest temporary directory fixture.

    Returns:
        Path: Path to the generated Parquet file containing video guidance.
    """
    guidance = pl.DataFrame(
        {
            "video_id": ["v1"],
            "target_frames": [[1, 2]],
            "bird": [1],
            "added_rests": [1],
            "video_path": ["2021_bunting_clips/test_video.mp4"],
            "fps": [10],
        }
    )
    path = tmp_path / "guidance.parquet"
    guidance.write_parquet(path)
    return path


class MockCapture:
    """Mock class for cv2.VideoCapture that returns predefined frames."""

    def __init__(self) -> None:
        """Initialize with two black frames of shape (100, 200, 3)."""
        self.frames: list[np.ndarray] = [
            np.zeros((100, 200, 3), dtype=np.uint8),
            np.zeros((100, 200, 3), dtype=np.uint8),
        ]
        self.index: int = 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the next frame or signal end of video.

        Returns:
            tuple: (success: bool, frame: np.ndarray or None)
        """
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def get(self, prop_id: int) -> int:
        """Mock the FPS retrieval.

        Args:
            prop_id (int): OpenCV property ID.

        Returns:
            int: Fixed FPS value (10).
        """
        return 10

    def release(self) -> None:
        """Mock release method (does nothing)."""
        pass


@patch("bird_cv.annotations_to_yolo.cv2.VideoCapture", return_value=MockCapture())
def test_process_item(
    mock_videocap, tmp_path: Path, sample_item: Dict[str, Any], path_to_guidance: Path
) -> None:
    """Test process_item writes one image and one label file per annotated frame.

    Args:
        mock_videocap: Mocked cv2.VideoCapture instance.
        tmp_path (Path): pytest temporary directory.
        sample_item (dict): Sample annotation item fixture.
        path_to_guidance (Path): Path to guidance Parquet fixture.
    """
    videos_dir = tmp_path / "videos"
    frames_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    videos_dir.mkdir()
    frames_dir.mkdir()
    labels_dir.mkdir()

    process_item(
        item=sample_item,
        path_to_videos=videos_dir,
        path_to_output_frames=frames_dir,
        path_to_output_labels=labels_dir,
        path_to_guidance=path_to_guidance,
    )

    # Two frames → two images + two label files
    image_files = list(frames_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))

    assert len(image_files) == 2
    assert len(label_files) == 2

    # Check label file format
    content = label_files[0].read_text().strip()
    assert content.startswith("0 ")
    assert len(content.split()) == 5


@patch("bird_cv.annotations_to_yolo.cv2.VideoCapture", return_value=MockCapture())
def test_stream_annotations_to_yolo(
    mock_videocap, tmp_path: Path, sample_item: Dict[str, Any], path_to_guidance: Path
) -> None:
    """Test stream_annotations_to_yolo processes a JSON file and writes YOLO labels.

    Runs in serial mode (processes=1) and verifies that both frames produce
    one image and one label file each.

    Args:
        mock_videocap: Mocked cv2.VideoCapture instance.
        tmp_path (Path): pytest temporary directory.
        sample_item (dict): Sample annotation item fixture.
        path_to_guidance (Path): Path to guidance Parquet fixture.
    """
    videos_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    videos_dir.mkdir()
    output_dir.mkdir()

    annotations_file = tmp_path / "annotations.json"

    # Write tiny JSON file with one item
    with open(annotations_file, "w") as f:
        json.dump([sample_item], f)

    # Run serially (processes=1)
    stream_annotations_to_yolo(
        path_to_videos=videos_dir,
        path_to_annotations=annotations_file,
        path_to_output=output_dir,
        path_to_guidance=path_to_guidance,
        processes=1,
    )

    frames_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    # Two frames → two images + two label files
    image_files = list(frames_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))

    assert len(image_files) == 2
    assert len(label_files) == 2

    # Check label file format
    content = label_files[0].read_text().strip()
    assert content.startswith("0 ")
    assert len(content.split()) == 5
