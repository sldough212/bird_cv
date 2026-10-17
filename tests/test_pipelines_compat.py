"""Guard: the existing Label Studio pipelines in bird_cv/pipelines/ keep working.

New pipeline work may add keyword arguments with defaults to the functions
these scripts call, but must not remove, reorder or newly require any
parameter (see docs/adr/0002-canonical-frame-index.md).
"""

import importlib
import inspect

import pytest

PIPELINE_MODULES = [
    "bird_cv.pipelines.classify",
    "bird_cv.pipelines.detect",
    "bird_cv.pipelines.retrain_classify",
    "bird_cv.pipelines.retrain_detect",
    "bird_cv.pipelines.temp_track_metrics",
    "bird_cv.pipelines.upload_for_classify",
    "bird_cv.pipelines.upload_for_detect",
    "bird_cv.pipelines.upload_for_retrain_detect",
]

# Parameters each function had when the new pipeline work began, in order.
EXPECTED_PARAMS = {
    "bird_cv.classification.train_video_model:train_video_model": [
        "clips_root",
        "output_root",
        "model_checkpoint",
        "output_name",
        "num_frames",
        "epochs",
        "batch_size",
        "lr",
        "device",
        "freeze_encoder",
    ],
    "bird_cv.client.evaluate_label_studio_tracking:evaluate_label_studio_tracking": [
        "host",
        "port",
        "api_key",
        "project_name",
        "output_path",
        "annotation_id",
        "prediction_id",
        "model_version",
    ],
    "bird_cv.client.get_label_studio_annotations:get_label_studio_annotations": [
        "host",
        "port",
        "api_key",
        "project_name",
        "output_path",
        "interpolate_frames",
        "snapshot_title",
    ],
    "bird_cv.client.post_bbox_predictions:post_bbox_predictions": [
        "host",
        "port",
        "api_key",
        "project_name",
        "model_path",
        "tracker_path",
        "model_version",
        "predict_labeled",
    ],
    "bird_cv.client.post_behavior_predictions:post_behavior_predictions": [
        "host",
        "port",
        "api_key",
        "project_name",
        "model_path",
        "model_version",
        "predict_labeled",
    ],
    "bird_cv.client.upload_tasks:upload_video_tasks": [
        "port",
        "host",
        "api_key",
        "project_name",
        "video_path",
    ],
    "bird_cv.detection.train_yolo:update_train_yolo": [
        "base_path",
        "previous_model_config",
        "pretrained_checkpoint",
        "yolo_training_data",
        "epochs",
        "device",
        "tune",
        "tune_iterations",
        "run_name",
    ],
    "bird_cv.pipelines.config:load_config": ["config_path", "config_type"],
    "bird_cv.preprocessing.annotations_to_mae:stream_annotations_to_mae": [
        "path_to_videos",
        "path_to_annotations",
        "path_to_guidance",
        "path_to_output",
        "num_frames",
        "stride",
        "processes",
    ],
    "bird_cv.preprocessing.annotations_to_yolo:stream_annotations_to_yolo": [
        "path_to_videos",
        "path_to_annotations",
        "path_to_output",
        "path_to_guidance",
        "processes",
        "crop_size",
    ],
    "bird_cv.preprocessing.crop:crop_cages": [
        "split_guidance_path",
        "video_segments_path",
        "clip_output_path",
        "videos_path",
    ],
    "bird_cv.preprocessing.get_split_guidance:simulate_split_guidance": [
        "videos_path",
        "output_path",
    ],
    "bird_cv.preprocessing.get_split_guidance:split_guidance_within_camera": [
        "path_to_guidance",
        "split_ratio",
        "output_path",
    ],
    "bird_cv.preprocessing.image_utils:run_images_to_video": [
        "split_guidance_path",
        "clip_output_path",
        "video_output_path",
    ],
    "bird_cv.segmentation.segment:run_segment": [
        "segmentation_configs_path",
        "sam_model_id",
        "split_guidance_path",
        "segmentations_path",
        "videos_path",
    ],
}


@pytest.mark.parametrize("module", PIPELINE_MODULES)
def test_pipeline_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("target", sorted(EXPECTED_PARAMS))
def test_pipeline_dependency_signature_is_compatible(target: str) -> None:
    module_name, func_name = target.split(":")
    params = list(
        inspect.signature(
            getattr(importlib.import_module(module_name), func_name)
        ).parameters.values()
    )
    expected = EXPECTED_PARAMS[target]

    assert [p.name for p in params[: len(expected)]] == expected
    for extra in params[len(expected) :]:
        assert extra.default is not inspect.Parameter.empty, (
            f"{target}: new parameter {extra.name!r} must have a default"
        )
