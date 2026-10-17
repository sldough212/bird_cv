"""End-to-end smoke test (ADR 0004): shape and frame alignment only, no scores."""

from pathlib import Path

import polars as pl
import pytest

from bird_cv.core.labels import CLASSES, NO_BIRD
from bird_cv.core.schemas import DETECTION_SCHEMA, PREDICTION_SCHEMA
from bird_cv.core.timing import StageTimer
from bird_cv.pipelines.config import InferConfig

ROOT = Path("/Users/sdougherty/Documents/code/bird_cv")
CLIP = ROOT / "scratch/20260627/videos/H7,I22/54/1.mp4"
YOLO = ROOT / "label-studio-ml-backend/label_studio_ml/examples/yolo/models/best.pt"
VIDEOMAE = ROOT / "my_ml_backend/models/best"
TRACKER = Path(__file__).parents[2] / "bird_cv/detection/tracker.yaml"
N_FRAMES = 30


@pytest.mark.slow
@pytest.mark.skipif(
    not (CLIP.exists() and YOLO.exists() and VIDEOMAE.exists()),
    reason="local smoke-test clip or checkpoints not available",
)
def test_e2e_smoke_on_single_cage_clip(tmp_path: Path) -> None:
    from bird_cv.classification.predict_video_model import VideoMAEClassifier
    from bird_cv.detection.track import YoloDetector, track_video
    from bird_cv.inference.cages import WholeFrameSegmenter
    from bird_cv.inference.pipeline import infer_video

    # The clip is already one cage; lay it out as {camera}/{video}.mp4
    raw = tmp_path / "raw" / "H7,I22" / "54.mp4"
    raw.parent.mkdir(parents=True)
    raw.symlink_to(CLIP)
    timer = StageTimer()
    detector = YoloDetector(model_path=YOLO, tracker_path=TRACKER)

    df = infer_video(
        raw,
        tmp_path / "run",
        WholeFrameSegmenter(),
        detector,
        VideoMAEClassifier(model_path=str(VIDEOMAE)),
        # Small chunks so chunking, tracker carry-over and stitching all run
        InferConfig(chunk_frames=12, chunk_overlap=4),
        timer,
        max_frames=N_FRAMES,
    )

    assert df.schema == pl.Schema(PREDICTION_SCHEMA)
    dets = pl.read_parquet(tmp_path / "run" / "detections" / "H7,I22" / "54.parquet")
    assert dets.schema == pl.Schema(DETECTION_SCHEMA) and dets.height > 0
    assert df["frame"].to_list() == list(range(N_FRAMES))
    assert set(df["cage_id"]) == {"0"} and set(df["camera_id"]) == {"H7,I22"}
    assert set(df["label"]) <= set(CLASSES)
    boxed = df.filter(pl.col("label") != NO_BIRD)
    assert boxed["bbox_left"].null_count() == 0
    assert df.filter(pl.col("label") == NO_BIRD)["bbox_left"].null_count() == (
        df.height - boxed.height
    )

    # Frame alignment: frames detected at det_conf keep exactly the box the
    # tracker gives when run on the whole standardized video in one go
    std = tmp_path / "run" / "standardized" / "H7,I22" / "54.mp4"
    reference = {d.frame: d for d in track_video(std, YOLO, TRACKER) if d.conf >= 0.25}
    assert reference, "expected some detections on the clip"
    for row in boxed.iter_rows(named=True):
        if row["frame"] in reference:
            ref = reference[row["frame"]]
            assert row["bbox_left"] == pytest.approx(ref.left, abs=1e-4)
            assert row["bbox_top"] == pytest.approx(ref.top, abs=1e-4)
            assert row["det_conf"] == pytest.approx(ref.conf, abs=1e-4)
    assert set(reference) <= set(boxed["frame"])

    stages = set(timer.to_frame()["stage"])
    assert stages == {"standardize", "sam2", "detect", "classify", "write"}
    print(df.group_by("label").len(), timer.to_frame())
