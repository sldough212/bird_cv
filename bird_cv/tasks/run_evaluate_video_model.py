from pathlib import Path

from bird_cv.classification.evaluate_video_model import evaluate_video_model

evaluate_video_model(
    clips_root=Path("/gscratch/pdoughe1/20260415_163412/video_crop"),
    model_path=Path(
        "/gscratch/pdoughe1/20260415_163412/training/videomae_behavior/best"
    ),
    output_path=Path("/gscratch/pdoughe1/20260415_163412/evaluation/videomae_behavior"),
)
