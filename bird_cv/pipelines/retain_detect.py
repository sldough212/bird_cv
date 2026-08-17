from bird_cv.preprocessing.get_split_guidance import split_guidance_within_camera
from bird_cv.preprocessing.annotations_to_yolo import stream_annotations_to_yolo
from bird_cv.detection.train_yolo import update_train_yolo
from bird_cv.pipelines.config import BirdCVConfig


def run_retrain_detect(cfg: BirdCVConfig) -> None:
    #####################################################################
    # How is this stored between detect and classify?
    #####################################################################
    split_guidance_within_camera(
        path_to_guidance=cfg.path_to_guidance,
        split_ratio={"train": 0.75, "val": 0.25},
        output_path=cfg.path_to_guidance_within_camera,
    )
    #####################################################################

    stream_annotations_to_yolo(
        path_to_videos=cfg.path_to_cage_videos,
        path_to_annotations=cfg.path_to_detect_annotations,
        path_to_guidance=cfg.path_to_guidance_within_camera,
        path_to_output=cfg.detect.path_to_training_data,
        processes=1,
        crop_size=None,
    )

    update_train_yolo(
        base_path=cfg.detect.path_to_output,
        previous_model_config=cfg.detect.path_to_yolo_config,
        pretrained_checkpoint=cfg.detect.path_to_yolo,
        yolo_training_data=cfg.detect.path_to_training_data,
        epochs=cfg.detect.epochs,
        device=cfg.detect.device,
        tune=cfg.detect.tune,
        tune_iterations=cfg.detect.tune_iterations,
        run_name=cfg.detect.run_name,  # is this really necessary?
    )
