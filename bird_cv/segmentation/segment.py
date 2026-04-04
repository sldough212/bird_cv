from pathlib import Path
import os
import json
import shutil
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# External / project-specific dependencies
from sam2.build_sam import build_sam2_video_predictor

# Local utilities (adjust module paths as needed)
from bird_cv.segmentation.visualize import (
    show_points,
    show_mask,
    vizualize_segmentations,
)
from bird_cv.segmentation.smooth import smooth_masks
from bird_cv.utils import NumpyEncoder


def get_camera_sam_config(
    model_checkpoint_path: Path,
    video_base_path: Path,
    camera_id: str,
    video_id: str,
    guess_prompts: dict[str, dict],
    ann_frame_idx: int = 0,
    winner: bool = False,
    output_config_path: Path | None = None,
    return_video_segments: bool = False,
    output_segment_path: Path | None = None,
) -> None:
    """Generates and optionally saves SAM2 segmentation prompts for a camera.

    This function loads a SAM2 model and applies a set of candidate segmentation
    prompts (points and labels) to a specific frame of a video. It visualizes the
    resulting masks and points for manual inspection, allowing the user to assess
    whether the prompts produce satisfactory segmentation.

    If the provided prompts are deemed correct (``winner=True``), the function
    saves them as a JSON configuration file for future use.

    Args:
        model_checkpoint_path (Path): Path to the directory containing the SAM2
            model checkpoint file.
        video_base_path (Path): Base directory containing video frame folders
            organized by camera and video IDs.
        camera_id (str): Identifier for the camera (used to locate the video
            directory and name the output file).
        video_id (str): Identifier for the specific video (subdirectory of the
            camera directory containing frames).
        guess_prompts (dict[str, dict]): Dictionary of candidate segmentation
            prompts. Each key represents an object (e.g., cage), and each value
            must contain:
                - "points": List of point coordinates.
                - "labels": List of labels corresponding to each point.
        ann_frame_idx (int, optional): Index of the frame on which to apply and
            visualize the prompts. Defaults to 0.
        winner (bool, optional): Whether the provided prompts are considered
            correct. If True, they will be saved to disk. Defaults to False.
        output_config_path (Path | None, optional): Directory where the configuration
            JSON file will be saved if ``winner`` is True. Required when
            ``winner=True``.
        return_video_segments (bool): Whether to return the segmentation arrays
            displayed when setting the initial SAM2 masks
        output_segment_path (Path | None, optional): Path to where segmentation masks
            are saved.

    Raises:
        ValueError: If ``winner`` is True but ``output_config_path`` is not provided.
            Likewise for return_video_segments and ``output_segment_path``
    """
    if winner and not output_config_path:
        raise ValueError(
            "output_config_path must be provided if guess prompt is the winner for that camera"
        )

    if return_video_segments and not output_segment_path:
        raise ValueError(
            "output_segment_path must be provided if wanting to return the segmentation masks"
        )

    # Load in model (always large for now)
    sam2_checkpoint = model_checkpoint_path / "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # We only need a cpu to load segment the very first frame
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device="cpu")

    # Load in data
    video_path = video_base_path / camera_id / video_id
    inference_state = predictor.init_state(video_path=str(video_path))

    # To be safe
    predictor.reset_state(inference_state)

    # Setup exploratory vizualization
    frame_names = [
        p for p in os.listdir(video_path) if os.path.splitext(p)[-1] in [".jpg"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_path, frame_names[ann_frame_idx])))

    for ii, (cage, prompt) in enumerate(guess_prompts.items()):
        ann_obj_id = ii
        points = prompt["points"]
        labels = prompt["labels"]

        # Attempt segmentation
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # Visualize the results
        show_points(points, labels, plt.gca())
        show_mask(
            (out_mask_logits[-1] > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id
        )

    if return_video_segments:
        assert output_segment_path is not None
        video_segments = {}
        video_segments[ann_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        save_video_segments(
            output_path=output_segment_path, video_segments=video_segments
        )

    if winner:
        assert output_config_path is not None
        # Save json config
        output_config_path.mkdir(exist_ok=True, parents=True)
        with open(output_config_path / f"{camera_id}.json", "w") as f:
            json.dump(guess_prompts, f, cls=NumpyEncoder)


def set_sam_and_predict(
    model_checkpoint_path: Path,
    data_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> dict[int, dict[int, np.ndarray]]:
    """Initializes a SAM2 video predictor, applies segmentation prompts, and
    generates mask predictions for objects in a video.

    Args:
        model_checkpoint_path (Path): Path to the directory containing the SAM2
            model checkpoint file (e.g., "sam2.1_hiera_large.pt").
        data_path (Path): Path to the input video or directory of frames to be
            processed by the predictor.
        config_path (Path): Path to a JSON file containing segmentation prompts.
            The file should map object identifiers (e.g., cage names) to a dict
            with keys:
                - "points": List of point coordinates for prompting.
                - "labels": List of labels corresponding to each point.
        device (str, optional): Device to run inference on (e.g., "cpu", "cuda").
            Defaults to "cpu".

    Returns:
        dict[int, dict[int, np.ndarray]]: A nested dictionary containing predicted
            masks. The outer dictionary maps frame indices to inner dictionaries,
            which map object IDs to their corresponding mask arrays (as NumPy arrays).
    """
    # Load the frames into the sam2 model
    sam2_checkpoint = model_checkpoint_path / "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=str(data_path))

    # Load in the config
    with config_path.open("r") as f:
        segmenation_config = json.load(f)

    # Now set the sam2 model
    for ii, (cage, prompt) in enumerate(segmenation_config.items()):
        ann_obj_id = ii
        points = prompt["points"]
        labels = prompt["labels"]

        # Attempt segmentation
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    # Let's try to display
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def save_video_segments(
    output_path: Path,
    video_segments: dict[int, dict[int, np.ndarray]],
    save_off_frame: bool = False,
) -> None:
    """Saves video segmentation results to disk in JSON format.

    Args:
        output_path (Path): Path to the output JSON file where segmentation
            results will be saved.
        video_segments (dict[int, dict[int, np.ndarray]]): Nested dictionary
            containing segmentation results. The outer dictionary maps frame
            indices to inner dictionaries, which map object IDs to their
            corresponding segmentation masks (NumPy arrays).
        save_off_frames (bool): Whether to instead save the mask of each frame separately
    """
    (output_path.parent).mkdir(exist_ok=True, parents=True)
    if save_off_frame:
        for frame_idx, frame_segments in video_segments.items():
            frame_output = output_path.parent / f"{frame_idx}_{output_path.name}"
            with frame_output.open("w") as f:
                json.dump(frame_segments, f, cls=NumpyEncoder)
    else:
        with output_path.open("w") as f:
            json.dump(video_segments, f, cls=NumpyEncoder)


def segment(
    config_path: Path,
    x0_frame_path: Path,
    y_video_path: Path,
    model_checkpoint_path: Path,
    output_path: Path,
    device: str = "cpu",
    visualize: bool = False,
    vis_frame_stride: int = 100,
) -> None:
    """Runs an end-to-end video segmentation pipeline using SAM2.

    Args:
        config_path (Path): Path to a JSON configuration file containing
            segmentation prompts (points and labels per object).
        x0_frame_path (Path): Path to the initial frame image used to seed
            segmentation (will be renamed to "00000.jpg").
        y_video_path (Path): Path to the input video from which frames will be
            extracted and segmented.
        model_checkpoint_path (Path): Path to the directory containing the SAM2
            model checkpoint.
        output_path (Path): Path where the segmentation results will be saved
            (JSON format).
        device (str, optional): Device to run inference on (e.g., "cpu", "cuda").
            Defaults to "cpu".
        visualize (bool): Whether to display the predicted segmentations.
        vis_frame_stride (int): If displaying the predicted segmentations, this
            parameter controls the stride between visualized frames.
    """

    # Get a directory to temporarily store the frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Move the x0_frame in the temperary directory
        shutil.copy(x0_frame_path, temp_path)

        # Rename this to frame 0
        shutil.move(temp_path / x0_frame_path.name, temp_path / "00000.jpg")

        # Extract the frames from the desired video into the same directory
        shutil.copytree(y_video_path, temp_path, dirs_exist_ok=True)

        # Set segmentation model and predict
        video_segments = set_sam_and_predict(
            model_checkpoint_path=model_checkpoint_path,
            data_path=temp_path,
            config_path=config_path,
            device=device,
        )

        # Smooth the SAM2 segmentaitons with a polygon approximation
        video_segments = smooth_masks(video_segments=video_segments)

        # Remove the first frame, this was the config frame
        video_segments.pop(0)
        (temp_path / "00000.jpg").unlink()

        # Move keys up
        video_segments = {ii: value for ii, value in enumerate(video_segments.values())}

        # Save the video segments
        save_video_segments(
            output_path=output_path,
            video_segments=video_segments,
            save_off_frame=True,
        )

        # Visualize segmentations
        if visualize:
            vizualize_segmentations(
                video_dir=temp_path,
                video_segments=video_segments,
                vis_frame_stride=vis_frame_stride,
            )
