from bird_cv.preprocessing.annotations_to_mae import stream_annotations_to_mae
from bird_cv.classification.train_video_model import train_video_model
from bird_cv.pipelines.config import BirdCVConfig


def run_retrain_classify(cfg: BirdCVConfig) -> None:
    # Crop behaviors from the annoations
    stream_annotations_to_mae(
        path_to_videos=cfg.path_to_bbox_videos,
        path_to_annotations=cfg.path_to_classify_annotations,
        path_to_guidance=cfg.path_to_guidance_within_camera,  # how will we get this here
        path_to_output=cfg.classify.path_to_training_data,
        processes=1,
    )

    train_video_model(
        clips_root=cfg.classify.path_to_training_data,
        output_root=cfg.classify.path_to_output,
        model_checkpoint=cfg.classify.path_to_mae,
        num_frames=cfg.classify.num_frames,
        epochs=cfg.classify.epochs,
        batch_size=cfg.classify.batch_size,
        lr=cfg.classify.lr,
        device=cfg.classify.device,
        # mode name ignored here
    )
