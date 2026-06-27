import cv2
import polars as pl
from pathlib import Path
from bird_cv.segmentation.segment import run_segment
from bird_cv.utils import extract_camera_video
from bird_cv.segmentation.frames import extract_all_frames
import json
import tempfile
from PIL import Image
from bird_cv.preprocessing.image_utils import crop_and_mask_image
from bird_cv.preprocessing.crop_yolo_labels import _load_cage_masks
import imageio_ffmpeg
import subprocess


def simulate_split_guidance(videos_path: Path, output_path: Path) -> None:
    """Build a split guidance parquet covering every frame of every video found under videos_path.

    Recursively scans for video files, reads their FPS and frame count via OpenCV,
    and writes a parquet with columns ``video_path``, ``fps``, and ``target_frames``
    (a list of all frame indices for each video). Intended as a stand-in for the
    real split guidance produced by the preprocessing pipeline when all frames
    should be processed.

    Args:
        videos_path: Root directory to search recursively for video files.
        output_path: Destination path for the output parquet file.
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    video_path_store, fps_store, target_frames_store = [], [], []
    for p in videos_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_extensions:
            # Determine fps of the video
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Find frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()

            video_path_store.append(str(p))
            fps_store.append(fps)
            target_frames_store.append(list(range(1, frame_count + 1)))

    split_guidance = pl.DataFrame(
        {
            "video_path": video_path_store,
            "fps": fps_store,
            "target_frames": target_frames_store,
        }
    )

    split_guidance.write_parquet(output_path)


def crop_cages(
    split_guidance_path: Path,
    video_segments_path: Path,
    clip_output_path: Path,
    videos_path: Path,
) -> None:
    """Crop each cage region from every target frame and save as individual JPEGs.

    For each video in the split guidance, extracts all frames to a temporary
    directory, loads the corresponding SAM2 cage masks, and saves a cropped
    JPEG per cage per frame under ``clip_output_path/{camera_id}/{video_id}/{cage_id}/``.
    Frames with missing segmentation masks are skipped.

    Args:
        split_guidance_path: Path to the split guidance parquet produced by
            ``simulate_split_guidance`` or the preprocessing pipeline.
        video_segments_path: Root directory of SAM2 segmentation JSON outputs,
            structured as ``{camera_id}/{video_id}/``.
        clip_output_path: Root directory where cropped cage images will be saved.
        videos_path: Root directory containing the source video files.
    """
    split_guidance = pl.read_parquet(split_guidance_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for video_str, target_frames in split_guidance.select(
            "video_path", "target_frames"
        ).to_numpy():
            camera_id, video_name = extract_camera_video(video_str=video_str)
            video_id = Path(video_name).stem
            image_output_path = clip_output_path / camera_id / video_id
            print(f"  Cropping {camera_id}/{video_id}")

            frame_store_path = temp_path / camera_id / video_id
            frame_store_path.mkdir(exist_ok=True, parents=True)
            extract_all_frames(
                video_path=videos_path / camera_id / video_name,
                output_path=frame_store_path,
            )

            # Extract segmentation info
            seg_dir = video_segments_path / camera_id / video_id
            seg_index_path = seg_dir / "segment_index.json"
            with seg_index_path.open("r") as f:
                segment_index = json.load(f)

            # Crop each cage in camera / video
            for frame in sorted(target_frames):
                frame_cage_masks = _load_cage_masks(seg_dir, segment_index, frame)
                if frame_cage_masks is None:
                    continue

                img = Image.open(frame_store_path / f"{frame:05d}.jpg")

                for cage_id, cage_mask in frame_cage_masks.items():
                    cropped_img, _ = crop_and_mask_image(
                        img, cage_mask, black_out=True, padding=5
                    )

                    crop_output = image_output_path / cage_id
                    crop_output.mkdir(exist_ok=True, parents=True)

                    cropped_img.save(crop_output / f"{frame:05d}.jpg")


def images_to_video(image_dir: Path, output_path: Path, fps: float = 30) -> None:
    """Encode a directory of JPEG frames into an H.264 MP4 using ffmpeg.

    Reads all ``*.jpg`` files from ``image_dir`` in sorted order and pipes
    them as raw BGR frames to ffmpeg. Frames are padded to even dimensions
    if necessary for H.264 compatibility.

    Args:
        image_dir: Directory containing sequentially named JPEG frames.
        output_path: Destination path for the output MP4 file.
        fps: Frame rate of the output video.

    Raises:
        FileNotFoundError: If no JPEG files are found in ``image_dir``.
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    frames = sorted(image_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No JPGs found in {image_dir}")

    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]

    # Round dimensions up to the nearest even number
    new_w = w + (w % 2)
    new_h = h + (h % 2)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{new_w}x{new_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        # Pad bottom/right by 1px if needed, to match new_w/new_h
        if frame.shape[1] != new_w or frame.shape[0] != new_h:
            frame = cv2.copyMakeBorder(
                frame,
                top=0,
                bottom=new_h - frame.shape[0],
                left=0,
                right=new_w - frame.shape[1],
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")


def run_images_to_video(
    split_guidance_path: Path,
    clip_output_path: Path,
    video_output_path: Path,
) -> None:
    """Convert cropped cage JPEG sequences into MP4 videos for every cage and video.

    Reads the split guidance to determine FPS per video, then iterates over
    all cage directories under ``clip_output_path/{camera_id}/{video_id}/``
    and encodes each into an MP4 at ``video_output_path/{camera_id}/{video_id}/{cage_id}.mp4``.

    Args:
        split_guidance_path: Path to the split guidance parquet supplying
            ``video_path`` and ``fps`` columns.
        clip_output_path: Root directory of cropped cage JPEG images produced
            by ``crop_cages``.
        video_output_path: Root directory where output MP4 files will be written.
    """
    split_guidance = pl.read_parquet(split_guidance_path)

    for video_str, fps in split_guidance.select("video_path", "fps").iter_rows():
        camera_id, video_name = extract_camera_video(video_str=video_str)
        video_id = Path(video_name).stem
        image_output_path = clip_output_path / camera_id / video_id

        # Iterate through the cages
        for cage_path in image_output_path.iterdir():
            if not cage_path.is_dir():
                continue

            cage_output_path = (
                video_output_path / camera_id / video_id / f"{cage_path.name}.mp4"
            )
            cage_output_path.parent.mkdir(exist_ok=True, parents=True)

            images_to_video(image_dir=cage_path, fps=fps, output_path=cage_output_path)


base_output = Path("/gscratch/pdoughe1/label_studio_test")
base_output.mkdir(exist_ok=True, parents=True)
intermediate_output = base_output / "intermediate"
intermediate_output.mkdir(exist_ok=True, parents=True)

videos_path = Path("/gscratch/pdoughe1/videos/label_studio_integration")
segmentation_configs_path = Path("/gscratch/pdoughe1/segmentation_configs/configs")
model_checkpoint_path = Path("/home/pdoughe1/sam2/checkpoints")

simulate_split_guidance(
    videos_path, output_path=intermediate_output / "split_guidance.parquet"
)

run_segment(
    segmentation_configs_path=segmentation_configs_path,
    model_checkpoint_path=model_checkpoint_path,
    split_guidance_path=intermediate_output / "split_guidance.parquet",
    segmentations_path=intermediate_output / "segmentations",
    videos_path=videos_path,
)

crop_cages(
    split_guidance_path=intermediate_output / "split_guidance.parquet",
    video_segments_path=intermediate_output / "segmentations",
    clip_output_path=base_output / "cropped_images",
    videos_path=videos_path,
)

# Save videos as mp4s
run_images_to_video(
    split_guidance_path=intermediate_output / "split_guidance.parquet",
    clip_output_path=base_output / "cropped_images",
    video_output_path=base_output / "cropped_videos",
)
