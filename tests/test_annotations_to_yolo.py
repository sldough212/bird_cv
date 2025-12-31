# test_annotations_to_yolo.py
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from bird_cv.annotations_to_yolo import process_item, stream_annotations_to_yolo

# Sample annotation item
SAMPLE_ITEM = {
    "data": {"video": "2021_bunting_clips/test_video.mp4"},
    "annotations": [
        {
            "result": [
                {
                    "id": "r1",
                    "value": {
                        "labels": ["bird"],
                        "sequence": [
                            {"frame": 0, "x": 10, "y": 20, "width": 30, "height": 40},
                            {"frame": 1, "x": 50, "y": 60, "width": 70, "height": 80},
                        ],
                    },
                }
            ]
        }
    ],
}


# Mock VideoCapture to return 2 fake frames
class MockCapture:
    def __init__(self):
        self.frames = [
            np.zeros((100, 200, 3), dtype=np.uint8),
            np.zeros((100, 200, 3), dtype=np.uint8),
        ]
        self.index = 0

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def release(self):
        pass


@patch("bird_cv.annotations_to_yolo.cv2.VideoCapture", return_value=MockCapture())
def test_process_item(mock_videocap, tmp_path: Path):
    """process_item writes one image and one label per frame."""
    videos_dir = tmp_path / "videos"
    frames_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    videos_dir.mkdir()
    frames_dir.mkdir()
    labels_dir.mkdir()

    process_item(
        item=SAMPLE_ITEM,
        path_to_videos=videos_dir,
        path_to_output_frames=frames_dir,
        path_to_output_labels=labels_dir,
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
def test_stream_annotations_to_yolo(mock_videocap, tmp_path: Path):
    """stream_annotations_to_yolo runs serially and writes images + labels."""
    videos_dir = tmp_path / "videos"
    output_dir = tmp_path / "output"
    videos_dir.mkdir()
    output_dir.mkdir()

    annotations_file = tmp_path / "annotations.json"

    # Write tiny JSON file with one item
    with open(annotations_file, "w") as f:
        json.dump([SAMPLE_ITEM], f)

    # Run serially (processes=1)
    stream_annotations_to_yolo(
        path_to_videos=videos_dir,
        path_to_annotations=annotations_file,
        path_to_output=output_dir,
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
