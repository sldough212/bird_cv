from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from bird_cv.detection.rf_detr import RfdetrDetector

TRACKER = Path(__file__).parents[2] / "bird_cv/detection/tracker.yaml"


class FakeRfdetr:
    """Two boxes per frame; only the more confident one should be tracked."""

    class_names = ["bird"]

    def predict(self, images, threshold, include_source_image):
        return [
            sv.Detections(
                xyxy=np.array([[0, 0, 5, 5], [10, 20, 30, 60]], dtype=float),
                confidence=np.array([0.3, 0.9]),
                class_id=np.array([0, 0]),
            )
            for _ in images
        ]


def test_tracks_top_box_as_normalized_detections() -> None:
    detector = RfdetrDetector(model_path="unused", tracker_path=TRACKER, batch_size=2)
    detector._model = FakeRfdetr()
    frames = [np.zeros((100, 50, 3), dtype=np.uint8)] * 3
    detections = detector.frame_tracker().track(frames, frame_offset=10)

    assert [d.frame for d in detections] == [10, 11, 12]
    assert {d.track_id for d in detections} == {1}
    for d in detections:
        assert (d.left, d.top, d.width, d.height) == pytest.approx(
            (0.2, 0.2, 0.4, 0.4), abs=0.01
        )
        assert d.label == "bird"
