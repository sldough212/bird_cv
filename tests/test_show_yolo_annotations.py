from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from bird_cv.show_yolo_annotations import (
    pick_random_frame,
    draw_yolo_annotations,
    show_annotated_frame,
)


def test_pick_random_frame(tmp_path: Path) -> None:
    """pick_random_frame returns a PNG from the directory."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "frame.png"
    Image.new("RGB", (100, 100)).save(img_path)

    result = pick_random_frame(images_dir)

    assert result == img_path


def test_draw_yolo_annotations(tmp_path: Path) -> None:
    """draw_yolo_annotations modifies the image in-place."""
    img = Image.new("RGB", (200, 200), color="black")

    label_file = tmp_path / "labels.txt"
    label_file.write_text("0 0.5 0.5 0.4 0.4\n")

    draw_yolo_annotations(img, label_file)

    # Image should no longer be all black after drawing
    pixels = list(img.getdata())
    assert any(pixel != (0, 0, 0) for pixel in pixels)


def test_show_annotated_frame(tmp_path: Path, monkeypatch) -> None:
    """show_annotated_frame runs end-to-end without error."""
    yolo_dir = tmp_path
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    img_path = images_dir / "frame_0001.png"
    Image.new("RGB", (300, 200), color="black").save(img_path)

    label_path = labels_dir / "frame_0001.txt"
    label_path.write_text("0 0.5 0.5 0.3 0.3\n")

    # Prevent matplotlib from opening a window
    monkeypatch.setattr(plt, "show", lambda: None)

    show_annotated_frame(
        path_to_yolo=yolo_dir,
        frame_name="frame_0001.png",
    )
