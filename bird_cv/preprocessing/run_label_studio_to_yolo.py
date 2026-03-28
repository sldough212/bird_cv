from pathlib import Path
import polars as pl
from bird_cv.preprocessing.get_label_tables import get_label_tables
from bird_cv.preprocessing.get_split_guidance import split_camera_data
from bird_cv.preprocessing.annotations_to_yolo import stream_annotations_to_yolo


def run_label_studio_to_yolo(
    label_paths: list[Path],
    output_path: Path,
    videos_path: Path,
    video_split_ratio: dict,
    random_seed: int = 42,
) -> None:
    """Convert deployed label annotations to yolo labels and images.

    Args:
        label_paths (list[Path]): List of paths hosting annotations and interpolated
            annotations from each labeler
        output_path (Path): Path to output store
        videos_path (Path): Path to videos (2021_bunting_clips/)
        video_split_ratio (dict): Dictionary containing splits (key) and proportion of
            camera data for each split (value)
        random_seed (int): Seed for random number generation to ensure reproducibility.
            Defaults to 42.
    """

    # Create an intermediate directory
    intermediate_path = output_path / "intermediate"
    intermediate_path.mkdir(exist_ok=True, parents=True)

    # Create the yolo store directory
    yolo_store_path = output_path / "yolo_store"
    yolo_store_path.mkdir(exist_ok=True, parents=True)

    video_store = []
    frame_store = []

    for ii, label_path in enumerate(label_paths):
        # Save tables
        get_label_tables(
            label_json_path=label_path / "annotations.json",
            output_dir=intermediate_path / f"input_{ii}",
        )

        # Load tables and store
        video = pl.read_ndjson(intermediate_path / f"input_{ii}" / "video_data.ndjson")
        frame = pl.read_ndjson(intermediate_path / f"input_{ii}" / "frame_data.ndjson")
        video_store.append(video)
        frame_store.append(frame)

    videos = pl.concat(video_store)
    frames = pl.concat(frame_store)

    videos.write_ndjson(intermediate_path / "video_data_full.ndjson")
    frames.write_ndjson(intermediate_path / "frame_data_full.ndjson")

    # Determine splitting guidance from concatenated table
    split_camera_data(
        video_data_path=intermediate_path / "video_data_full.ndjson",
        frame_data_path=intermediate_path / "frame_data_full.ndjson",
        split_ratio=video_split_ratio,
        output_path=intermediate_path,
        random_seed=random_seed,
    )

    # Move the labels and frames to prepare for yolo training
    for ii, label_path in enumerate(label_paths):
        stream_annotations_to_yolo(
            path_to_videos=videos_path,
            path_to_annotations=label_path / "annotations_interpolated.json",
            path_to_output=yolo_store_path,
            path_to_guidance=intermediate_path / "split_guidance.parquet",
            processes=1,
        )
