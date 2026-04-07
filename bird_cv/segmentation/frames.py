from pathlib import Path
import cv2


def extract_all_frames(
    video_path: Path, output_path: Path, frames: list[int] | None = None
) -> None:
    """Extracts all frames from a video file and saves them as images.

    Args:
        video_path (Path): Path to the input video file to be processed.
        output_path (Path): Directory where extracted frames will be saved.
        frames (list[int] | optional): List of specific frames to only extract.
    """
    output_path.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frames is not None and frame_count not in frames:
            frame_count += 1
            continue

        frame_count += 1
        out_path = output_path / f"{frame_count:05d}.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()
