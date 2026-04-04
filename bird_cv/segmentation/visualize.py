from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

import cv2
import subprocess
import json


def vizualize_segmentations(
    video_dir: Path,
    video_segments: dict[int, dict[int, np.ndarray]],
    vis_frame_stride: int = 100,
) -> None:
    """Visualizes segmentation masks overlaid on video frames.

    Args:
        video_dir (Path): Path to the directory containing video frames as image
            files (e.g., .jpg or .jpeg). Filenames are expected to be numeric so
            they can be sorted in chronological order.
        video_segments (dict[int, dict[int, np.ndarray]]): Nested dictionary
            containing segmentation results. The outer dictionary maps frame
            indices to inner dictionaries, which map object IDs to their
            corresponding segmentation masks (NumPy arrays).
    """
    # Scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Render the segmentation results every few frames
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


# SAM utils ###############################################################
def show_mask(
    mask: np.ndarray, ax: Axes, obj_id: int | None = None, random_color: bool = False
) -> None:
    """Overlay a segmentation mask on a Matplotlib axis with optional coloring.

    The function visualizes a 2D binary mask by applying a semi-transparent color
    overlay. The color can be selected based on an object ID or randomly.

    Args:
        mask (np.ndarray): 2D binary mask of shape (height, width), where non-zero
            values indicate the object region.
        ax (matplotlib.axes.Axes): Matplotlib axis on which to display the mask.
        obj_id (int, optional): Object identifier used to select a consistent color
            from the tab10 colormap. Defaults to None.
        random_color (bool, optional): If True, generates a random color for the mask
            instead of using the colormap. Defaults to False.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(
    coords: np.ndarray, labels: np.ndarray, ax: Axes, marker_size: int = 200
) -> None:
    """Display positive and negative points on a Matplotlib axis using color-coded markers.

    Args:
        coords (np.ndarray): Array of shape (N, 2) containing (x, y) coordinates of points.
        labels (np.ndarray): Array of shape (N,) containing binary labels for each point.
            - 1 indicates a positive point
            - 0 indicates a negative point
        ax (matplotlib.axes.Axes): Matplotlib axis on which to plot the points.
        marker_size (int, optional): Size of the markers. Defaults to 200.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box: np.ndarray, ax: Axes) -> None:
    """Draw a rectangular bounding box on a Matplotlib axis.

    Args:
        box (list or np.ndarray): List or array of four numbers [x_min, y_min, x_max, y_max]
            representing the top-left and bottom-right corners of the box.
        ax (matplotlib.axes.Axes): Matplotlib axis on which to draw the rectangle.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


###############################################################


def annotate_and_copy(image_path: Path):
    """Annotate points on an image interactively and copy the coordinates to the clipboard.

    This function opens the specified image in a window where the user can click to
    add points. Each click records the (x, y) coordinates of the point. Pressing 'q'
    or 'ESC' closes the window and copies all collected points to the system clipboard
    in JSON format. The function also returns the list of points.

    Args:
        image_path (Path): Path to the image file to annotate.

    Returns:
        list[list[int]]: A list of points annotated by the user. Each point is a list
        of two integers [x, y].

    Raises:
        ValueError: If the image could not be loaded from `image_path`.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    window_name = "Click = add | q/ESC = quit (copies all)"

    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = [x, y]
            points.append(point)

            print(f"Added: {point}")

            # Draw point
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, img)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Convert to JSON-style string
    points_str = json.dumps(points)

    # Copy to clipboard (macOS)
    try:
        subprocess.run("pbcopy", input=points_str.encode(), check=True)
        print(f"Copied to clipboard: {points_str}")
    except Exception as e:
        print("Clipboard copy failed:", e)

    return points


def visualize_predictions_over_ground_truth(
    segmentation_path: Path,
    camera_id: str,
    video_id: str,
    frames: list = [0, 1],
    vis_frame_stride: int = 100,
) -> None:
    """Visualizes masked predictions over ground truth segmentation.

    Args:
        segmentation_path (Path): Path to the directory containing prediction
            segmentation JSON files (e.g., camera_id/video_id/segmentation.json).
        camera_id (str): Camera ID to visualize.
        video_id (str): Video ID to visualize.
        frames (list): List of frame indices to visualize. Defaults to [0, 1].
        vis_frame_stride (int): Stride for rendering frames. Defaults to 100.
    """
    # Load prediction segmentation
    pred_segmentation_path = (
        segmentation_path / "predictions" / camera_id / video_id / "segmentation.json"
    )
    if not pred_segmentation_path.exists():
        raise FileNotFoundError(
            f"Prediction segmentation not found at: {pred_segmentation_path}"
        )

    with pred_segmentation_path.open("r") as f:
        pred_segmentation = json.load(f)

    # Load ground truth segmentation
    gt_segmentations = {}
    for frame in frames:
        gt_segmentation_path = (
            segmentation_path / "labels" / camera_id / f"{video_id}_{frame}.json"
        )
        if not gt_segmentation_path.exists():
            print(
                f"Warning: Ground truth not found for frame {frame}: {gt_segmentation_path}"
            )
            continue

        with gt_segmentation_path.open("r") as f:
            seg = json.load(f)
            key = next(iter(seg.keys()))
            value = next(iter(seg.values()))
            gt_segmentations[key] = value

    # Create a combined visualization dictionary
    # We'll map frame indices to a dictionary of object IDs and their masks
    # For predictions, frame indices are (int(frame)+1)
    # For ground truth, frame indices are just frame
    combined_segmentations = {}

    # Add predictions
    for frame in frames:
        frame = int(frame)
        pred_frame_key = str(frame + 1)
        if pred_frame_key in pred_segmentation:
            combined_segmentations[frame] = pred_segmentation[pred_frame_key]

    # Add ground truth (with a prefix to distinguish them)
    for frame, gt_data in gt_segmentations.items():
        frame = int(frame)
        if frame not in combined_segmentations:
            combined_segmentations[frame] = {}
        # We'll mark ground truth objects with a special prefix
        for obj_id, mask in gt_data.items():
            combined_segmentations[frame][f"gt_{obj_id}"] = mask

    # Find the directory containing video frames
    # Frames are expected to be in the same structure as predictions
    # Assuming frames are in segmentation_path / camera_id / video_id / frames/
    video_dir = segmentation_path / "frames" / camera_id / video_id

    if not video_dir.exists():
        print(f"Warning: Video frames directory not found: {video_dir}")
        return

    # Scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Render the segmentation results every few frames
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        if out_frame_idx not in combined_segmentations:
            continue

        plt.figure(figsize=(8, 6))
        plt.title(f"frame {out_frame_idx} | Camera: {camera_id} | Video: {video_id}")

        # Load and display the frame
        frame_path = video_dir / frame_names[out_frame_idx]
        if frame_path.exists():
            plt.imshow(Image.open(frame_path))
        else:
            print(f"Warning: Frame not found: {frame_path}")
            continue

        # Display predictions and ground truth (ground truth plotted first for lower z-order)
        # Separate ground truth and predictions
        gt_masks = []
        pred_masks = []

        for obj_id, mask in combined_segmentations[out_frame_idx].items():
            # Determine color first
            if obj_id.startswith("gt_"):
                # Ground truth - use a different color (bright red)
                color = np.array([1.0, 0.0, 0.0, 0.6])  # Bright red with transparency
            else:
                # Prediction - use different shades of blue
                cmap = plt.get_cmap("tab10")
                # Extract numeric part from obj_id for consistent colors
                obj_num = int(obj_id.split("_")[-1]) if "_" in str(obj_id) else 0
                color = np.array([*cmap(obj_num)[:3], 0.6])

            # Create mask image after color is determined
            mask_array = np.array(mask)
            h, w = mask_array.shape[-2:]
            mask_image = mask_array.reshape(h, w, 1) * color.reshape(1, 1, -1)

            if obj_id.startswith("gt_"):
                gt_masks.append(mask_image)
            else:
                pred_masks.append(mask_image)

        # Plot ground truth first (lowest z-order), then predictions
        for mask in gt_masks:
            plt.imshow(mask)
        for mask in pred_masks:
            plt.imshow(mask)

        plt.axis("off")
        plt.tight_layout()
        plt.show()
