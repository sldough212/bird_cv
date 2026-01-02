from pathlib import Path
import random
import matplotlib.pyplot as plt
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


def pick_random_frame(path_to_frames: Path) -> Path:
    """Randomly select a frame image from a directory.

    This function searches the provided directory for PNG image files
    and returns one at random. It raises an error if no matching files
    are found.

    Args:
        path_to_frames (Path): Path to a directory containing frame images
            saved as `.png` files.

    Returns:
        Path: Path to a randomly selected frame image.

    Raises:
        RuntimeError: If no `.png` files are found in the directory.
    """
    frame_files = list(path_to_frames.glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"No videos found in {path_to_frames}")
    return random.choice(frame_files)


def draw_yolo_annotations(frame: Image, label_file: Path) -> None:
    """Draw YOLO-format bounding boxes on a PIL image.

    This function reads a YOLO label file containing normalized bounding
    box coordinates and overlays the corresponding bounding boxes and
    class labels onto the provided PIL image in-place.

    The YOLO label format is expected to be:
        class_id x_center y_center width height

    All coordinates are assumed to be normalized to the range [0, 1].

    Args:
        frame (Image): A PIL Image object representing the frame on which
            annotations will be drawn.
        label_file (Path): Path to a YOLO label file (`.txt`) corresponding
            to the image.

    Returns:
        None
    """
    draw = ImageDraw.Draw(frame)
    frame_w, frame_h = frame.size

    # Optional: try to load a default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    with open(label_file, "r") as f:
        for line in f:
            class_id, x_c, y_c, w, h = map(float, line.split())

            # Convert normalized YOLO coords → pixel coords
            x_center = x_c * frame_w
            y_center = y_c * frame_h
            box_w = w * frame_w
            box_h = h * frame_h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w - 1, x2)
            y2 = min(frame_h - 1, y2)

            # Draw bounding box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="lime",
                width=4,
            )

            # Draw label
            label_text = f"class {int(class_id)}"
            text_pos = (x1, max(0, y1 - 12))

            if font:
                draw.text(text_pos, label_text, fill="lime", font=font)
            else:
                draw.text(text_pos, label_text, fill="lime")


def show_annotated_frame(
    path_to_yolo: Path,
    frame_name: Optional[str] = None,
) -> None:
    """Display a frame with YOLO annotations in a Jupyter notebook.

    This function loads a frame image and its corresponding YOLO label
    file, overlays bounding box annotations, and displays the result
    inline using matplotlib. If no frame name is provided, a random
    frame is selected.

    The expected directory structure is:
        path_to_yolo/
            ├── images/
            │   └── <frame_name>.png
            └── labels/
                └── <frame_name>.txt

    Args:
        path_to_yolo (Path): Path to the root YOLO dataset directory
            containing `images/` and `labels/` subdirectories.
        frame_name (Optional[str]): Name of the frame image file to
            display (e.g., `"video_frame_0001.png"`). If None, a random
            frame is selected.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified frame file does not exist.
    """
    path_to_labels = path_to_yolo / "labels"
    path_to_frames = path_to_yolo / "images"

    # Select Frame
    frame_path = (
        path_to_frames / frame_name if frame_name else pick_random_frame(path_to_frames)
    )
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    frame_stem = frame_path.stem

    # Load frame and label
    img = Image.open(frame_path)
    label_file = path_to_labels / f"{frame_stem}.txt"

    # Draw annotations
    draw_yolo_annotations(img, label_file)

    # Convert BGR → RGB for matplotlib
    img = img.convert("RGB")
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(frame_stem)
    plt.show()
