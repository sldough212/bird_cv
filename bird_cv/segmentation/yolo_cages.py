"""The ``yolo_cages`` segmenter: cages from a trained model, no prompts (ADR 0007)."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from bird_cv.core.ids import cage_id_from_index
from bird_cv.preprocessing.standardize import iter_canonical_frames, video_header
from bird_cv.segmentation.cage_masks import static_cage_masks


class YoloCageSegmenter:
    """Static cage masks from a trained YOLO segmentation model (ADR 0007).

    Cages never move within a video, so the model runs on ``sample_frames``
    evenly spaced frames only, and each cage gets one majority-vote mask
    (:func:`~bird_cv.segmentation.cage_masks.static_cage_masks`) for the
    whole video, numbered row-major.

    Args:
        model: The ``[models.cage_detector]`` model.
        conf: Detection confidence threshold.
        sample_frames: Frames to detect cages on.
        expected_cages: Cages to keep (the most confident); None keeps all.
    """

    def __init__(
        self,
        model: YOLO,
        conf: float = 0.5,
        sample_frames: int = 5,
        expected_cages: int | None = None,
    ) -> None:
        self.model = model
        self.conf = conf
        self.sample_frames = sample_frames
        self.expected_cages = expected_cages

    def segment_video(
        self, video_path: Path, chunk_frames: int
    ) -> list[tuple[int, dict[str, np.ndarray]]]:
        """One keyframe at frame 0 holding every cage's static mask.

        Raises:
            ValueError: If no cage, or fewer than ``expected_cages``, is found.
        """
        n = video_header(video_path)[1]
        wanted = {round(f) for f in np.linspace(0, max(n - 1, 0), self.sample_frames)}
        frames = [frame for _, frame in iter_canonical_frames(video_path, wanted)]
        if not frames:
            return []
        results = self.model.predict(
            frames, conf=self.conf, retina_masks=True, verbose=False
        )
        detections = [
            []
            if r.masks is None
            else list(zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.conf.tolist()))
            for r in results
        ]
        masks = static_cage_masks(detections, self.expected_cages)
        return [(0, {cage_id_from_index(i): m for i, m in enumerate(masks)})]
