from pathlib import Path

from bird_cv.segmentation.evaluate import predict_and_evaluate

TEST_PATH = Path("/Users/sdougherty/Documents/code/data/segmentation_test/test")
SEGMENTATION_CONFIG_PATH = Path(
    "/Users/sdougherty/Documents/code/data/segmentation_test/configs/configs"
)
VIDEO_BASE_PATH = Path(
    "/Users/sdougherty/Documents/code/data/segmentation_test/configs/frames"
)
MODEL_CHECKPOINT_PATH = Path("/Users/sdougherty/Documents/code/sam2/checkpoints")
PREDICTION_OUTPUT_PATH = TEST_PATH / "predictions"

predict_and_evaluate(
    test_path=TEST_PATH,
    segmentation_config_path=SEGMENTATION_CONFIG_PATH,
    video_base_path=VIDEO_BASE_PATH,
    model_checkpoint_path=MODEL_CHECKPOINT_PATH,
    prediction_output_path=PREDICTION_OUTPUT_PATH,
    output_path=TEST_PATH / "segment_evaluation.parquet",
)
