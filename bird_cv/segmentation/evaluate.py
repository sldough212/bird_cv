from pathlib import Path
import polars as pl
import numpy as np
import json

from bird_cv.segmentation.segment import segment
from bird_cv.segmentation.utils import calculate_iou


def predict_and_evaluate(
    test_path: Path,
    segmentation_config_path: Path,
    video_base_path: Path,
    model_checkpoint_path: Path,
    prediction_output_path: Path,
    output_path: Path,
) -> None:
    """
    Run segmentation predictions on a test dataset and evaluate their performance.

    Args:
        test_path (Path): Path to the test dataset. Should contain subdirectories:
            - `labels/` with ground-truth segmentation masks
            - `frames/` with input video frames
        segmentation_config_path (Path): Path to the directory containing per-camera segmentation
            configuration JSON files.
        video_base_path (Path): Path to the base directory containing training videos organized
            by camera.
        model_checkpoint_path (Path): Path to the trained model checkpoint used for prediction.
        prediction_output_path (Path): Path to the directory where prediction JSON files should
            be stored.
        output_path (Path): Path to the output file where the consolidated evaluation results
            (Parquet format) will be written.
    """

    # labels to evaluate against are in test_path / labels
    # frames to predict on are in test_path / frames
    evaluations = []
    for camera in (test_path / "labels").glob("*/"):
        camera_id = camera.stem
        print(f"Running prediction and evaluation on camera {camera_id}")
        for video in camera.glob("*"):
            video_id = video.stem[:-2]

            # Predict
            train_video_id = next(iter((video_base_path / camera_id).glob("*/"))).name

            # Skip if train video id is the same as the predict video id
            if video_id == train_video_id:
                continue

            # Skip if prediction already exists
            camera_video_pred_path = (
                prediction_output_path / camera_id / video_id / "segmentation.json"
            )
            if not camera_video_pred_path.exists():
                segment(
                    config_path=segmentation_config_path / f"{camera_id}.json",
                    x0_frame_path=video_base_path
                    / camera_id
                    / train_video_id
                    / "00001.jpg",
                    y_video_path=test_path / "frames" / camera_id / video_id,
                    model_checkpoint_path=model_checkpoint_path,
                    output_path=prediction_output_path
                    / camera_id
                    / video_id
                    / "segmentation.json",
                    device="cpu",
                    visualize=False,
                )

            # Evaluate
            evaluation = evaluate_segmentation(
                prediction_output_path=prediction_output_path,
                test_label_path=test_path / "labels",
                camera_id=camera_id,
                video_id=video_id,
            )
            evaluations.append(evaluation)

    # Save
    output = pl.concat(evaluations, how="diagonal_relaxed")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output.write_parquet(output_path)


def evaluate_segmentation(
    prediction_output_path: Path,
    test_label_path: Path,
    camera_id: str,
    video_id: str,
    frames: list = [0, 1],
) -> pl.DataFrame:
    """Evaluate segmentation predictions against ground-truth labels for a specific video.

    Args:
        prediction_output_path (Path): Path to the directory containing predicted segmentations
            in JSON format, organized by camera and video.
        test_label_path (Path): Path to the directory containing ground-truth labels in JSON format,
            organized by camera and frame.
        camera_id (str): Identifier for the camera to evaluate.
        video_id (str): Identifier for the video to evaluate.
        frames (list, optional): List of frame indices to evaluate. Defaults to [0, 1].

    Returns:
        pl.DataFrame: A Polars DataFrame containing the evaluation results with columns:
            - "camera_id": Camera identifier
            - "video_id": Video identifier
            - "cage": Cage identifier within the frame
            - "frames": Frame index
            - "iou": Intersection over Union score for the cage in that frame
    """

    # Load in both segments and determine the iou
    with (prediction_output_path / camera_id / video_id / "segmentation.json").open(
        "r"
    ) as f:
        pred_segmentation = json.load(f)

    test_segmentation = {}
    for frame in frames:
        with (test_label_path / camera_id / f"{video_id}_{frame}.json").open("r") as f:
            seg = json.load(f)
            key = next(iter(seg.keys()))
            value = next(iter(seg.values()))
            test_segmentation[key] = value

    cage_ids, frame_ids, ious = [], [], []
    for frame in frames:
        pred_cage = pred_segmentation[str(int(frame))]
        test_cage = test_segmentation[str(frame)]
        for cage in pred_cage.keys():
            iou = calculate_iou(
                pred_mask=np.squeeze(pred_cage[cage]),
                gt_mask=np.squeeze(test_cage[cage]),
            )
            cage_ids.append(cage)
            frame_ids.append(frame)
            ious.append(iou)

    evaluation = pl.DataFrame(
        {
            "camera_id": [camera_id] * len(cage_ids),
            "video_id": [video_id] * len(cage_ids),
            "cage": cage_ids,
            "frames": frame_ids,
            "iou": ious,
        }
    )

    return evaluation
