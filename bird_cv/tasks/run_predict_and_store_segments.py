from pathlib import Path
import polars as pl
import tempfile
from bird_cv.segmentation.frames import extract_all_frames
from bird_cv.segmentation.segment import segment
from bird_cv.utils import extract_camera_video


def predict_and_store_segments(
    split_guidance_path: Path,
    video_path: Path,
    prediction_output_path: Path,
    segmentation_config_path: Path,
    model_checkpoint_path: Path,
) -> None:
    # Load in split guidance with fps corrected target frames
    split_guidance = pl.read_parquet(split_guidance_path)

    # In a temporary directory, move the targeted frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for video_str, target_frames in split_guidance.select(
            "video_path", "target_frames"
        ).to_numpy():
            camera_id, video_name = extract_camera_video(video_str=video_str)
            video_id = Path(video_name).stem

            print(f"Segmenting, camera_id: {camera_id}, video_id: {video_name}")

            # Extract targeted frames
            frame_store = temp_path / camera_id / video_id
            frame_store.mkdir(exist_ok=True, parents=True)
            extract_all_frames(
                video_path=video_path / camera_id / video_name,
                output_path=frame_store,
                frames=target_frames,
            )

            # Run the segmentation
            train_video_id = next(
                iter((segmentation_config_path / "frames" / camera_id).glob("*/"))
            ).name

            # Skip if prediction already exists
            camera_video_pred_path = (
                prediction_output_path / camera_id / video_id / "segmentation.json"
            )
            if not camera_video_pred_path.exists():
                segment(
                    config_path=segmentation_config_path
                    / "configs"
                    / f"{camera_id}.json",
                    x0_frame_path=segmentation_config_path
                    / "frames"
                    / camera_id
                    / train_video_id
                    / "00001.jpg",
                    y_video_path=frame_store,
                    model_checkpoint_path=model_checkpoint_path,
                    output_path=prediction_output_path
                    / camera_id
                    / video_id
                    / "segmentation.json",
                    device="cuda",
                    visualize=False,
                )


if __name__ == "__main__":
    predict_and_store_segments(
        split_guidance_path=Path(
            "/gscratch/pdoughe1/20260411_161837/intermediate/split_guidance.parquet"
        ),
        video_path=Path("/gscratch/pdoughe1/videos/2021_bunting_clips"),
        prediction_output_path=Path("/gscratch/pdoughe1/20260411_161837/segmentations"),
        segmentation_config_path=Path(
            "/gscratch/pdoughe1/segmentation_configs/configs"
        ),
        model_checkpoint_path=Path("/home/pdoughe1/sam2/checkpoints"),
    )
