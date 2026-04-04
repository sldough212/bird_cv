from pathlib import Path
import cv2


def extract_first_frame(
    video_base_path: Path,
    output_path: Path,
    max_frames=100,
) -> None:
    """Extracts frames (up to a limit) from the first video in each subdirectory.

    The output structure is:
        output_path/
            <camera>/
                <video_name>/
                    00001.jpg
                    00002.jpg
                    ...

    Args:
        video_base_path (Path): Path to the base directory containing
            subdirectories of videos (e.g., one directory per camera).
        output_path (Path): Path to the directory where extracted frames will
            be saved. Subdirectories will be created automatically.
        max_frames (int, optional): Maximum number of frames to extract from
            each selected video. Defaults to 100.
    """
    video_exts = {
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".ts",
        ".mts",
    }

    for dr in video_base_path.iterdir():
        camera = dr.name

        if dr.is_dir():
            for video in dr.iterdir():
                if video.suffix.lower() in video_exts:
                    (output_path / camera / video.stem).mkdir(
                        exist_ok=True, parents=True
                    )

                    # Save then break
                    frame_count = 0
                    while True:
                        frame_count += 1
                        if frame_count > max_frames:
                            break

                        cap = cv2.VideoCapture(str(video))
                        success, frame = cap.read()
                        if success:
                            out_path = (
                                output_path
                                / camera
                                / video.stem
                                / f"{frame_count:05d}.jpg"
                            )
                            cv2.imwrite(out_path, frame)
                            # print(f"Saved: {out_path}")
                        else:
                            print(f"Failed to read: {video}")
                            break

                    cap.release()

                    break


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

        if frame_count not in frames:
            frame_count += 1
            continue

        frame_count += 1
        out_path = output_path / f"{frame_count:05d}.jpg"
        cv2.imwrite(str(out_path), frame)

    cap.release()
