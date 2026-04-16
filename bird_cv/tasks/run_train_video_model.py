from pathlib import Path

from bird_cv.classification.train_video_model import train_video_model

train_video_model(
    clips_root=Path("/gscratch/pdoughe1/20260415_163412/video_crop"),
    output_root=Path("/gscratch/pdoughe1/20260415_163412/training"),
    output_name="videomae_behavior",
    model_checkpoint="MCG-NJU/videomae-base",
    num_frames=16,
    epochs=10,
    batch_size=8,
    freeze_encoder=True,
)
