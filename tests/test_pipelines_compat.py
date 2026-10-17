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

# (name, repr(default)) each function had when the new pipeline work began,
# in order; None means no default.
EXPECTED_PARAMS: dict[str, list[tuple[str, str | None]]] = {
    "bird_cv.classification.train_video_model:train_video_model": [
        ("clips_root", None),
        ("output_root", None),
        ("model_checkpoint", None),
        ("output_name", "'videomae_behavior'"),
        ("num_frames", "16"),
        ("epochs", "10"),
        ("batch_size", "8"),
        ("lr", "0.0001"),
        ("device", "'cuda'"),
        ("freeze_encoder", "True"),
    ],
    "bird_cv.client.evaluate_label_studio_tracking:evaluate_label_studio_tracking": [
        ("host", None),
        ("port", None),
        ("api_key", None),
        ("project_name", None),
        ("output_path", None),
        ("annotation_id", "None"),
        ("prediction_id", "None"),
        ("model_version", "None"),
    ],
    "bird_cv.client.get_label_studio_annotations:get_label_studio_annotations": [
        ("host", None),
        ("port", None),
        ("api_key", None),
        ("project_name", None),
        ("output_path", None),
        ("interpolate_frames", "True"),
        ("snapshot_title", "None"),
    ],
    "bird_cv.client.post_bbox_predictions:post_bbox_predictions": [
        ("host", None),
        ("port", None),
        ("api_key", None),
        ("project_name", None),
        ("model_path", None),
        ("tracker_path", None),
        ("model_version", "None"),
        ("predict_labeled", "False"),
    ],
    "bird_cv.client.post_behavior_predictions:post_behavior_predictions": [
        ("host", None),
        ("port", None),
        ("api_key", None),
        ("project_name", None),
        ("model_path", None),
        ("model_version", "None"),
        ("predict_labeled", "False"),
    ],
    "bird_cv.client.upload_tasks:upload_video_tasks": [
        ("port", None),
        ("host", None),
        ("api_key", None),
        ("project_name", None),
        ("video_path", None),
    ],
    "bird_cv.detection.train_yolo:update_train_yolo": [
        ("base_path", None),
        ("previous_model_config", None),
        ("pretrained_checkpoint", None),
        ("yolo_training_data", None),
        ("epochs", "30"),
        ("device", "0"),
        ("tune", "False"),
        ("tune_iterations", "30"),
        ("run_name", "'bird_yolo'"),
    ],
    "bird_cv.pipelines.config:load_config": [
        ("config_path", None),
        ("config_type", "<class 'bird_cv.pipelines.config.BirdCVConfig'>"),
    ],
    "bird_cv.preprocessing.annotations_to_mae:stream_annotations_to_mae": [
        ("path_to_videos", None),
        ("path_to_annotations", None),
        ("path_to_guidance", None),
        ("path_to_output", None),
        ("num_frames", "16"),
        ("stride", "8"),
        ("processes", "4"),
    ],
    "bird_cv.preprocessing.annotations_to_yolo:stream_annotations_to_yolo": [
        ("path_to_videos", None),
        ("path_to_annotations", None),
        ("path_to_output", None),
        ("path_to_guidance", None),
        ("processes", "4"),
        ("crop_size", "None"),
    ],
    "bird_cv.preprocessing.crop:crop_cages": [
        ("split_guidance_path", None),
        ("video_segments_path", None),
        ("clip_output_path", None),
        ("videos_path", None),
    ],
    "bird_cv.preprocessing.get_split_guidance:simulate_split_guidance": [
        ("videos_path", None),
        ("output_path", None),
    ],
    "bird_cv.preprocessing.get_split_guidance:split_guidance_within_camera": [
        ("path_to_guidance", None),
        ("split_ratio", None),
        ("output_path", None),
    ],
    "bird_cv.preprocessing.image_utils:run_images_to_video": [
        ("split_guidance_path", None),
        ("clip_output_path", None),
        ("video_output_path", None),
    ],
    "bird_cv.segmentation.segment:run_segment": [
        ("segmentation_configs_path", None),
        ("sam_model_id", None),
        ("split_guidance_path", None),
        ("segmentations_path", None),
        ("videos_path", None),
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

    assert [
        (p.name, None if p.default is inspect.Parameter.empty else repr(p.default))
        for p in params[: len(expected)]
    ] == expected
    for extra in params[len(expected) :]:
        assert extra.default is not inspect.Parameter.empty, (
            f"{target}: new parameter {extra.name!r} must have a default"
        )
