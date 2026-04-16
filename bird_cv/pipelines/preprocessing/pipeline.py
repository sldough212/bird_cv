"""Preprocessing pipeline: label studio → YOLO → segmentation → crop → clips."""

from pathlib import Path

import msgspec
import polars as pl

from bird_cv.classification.build_clip_index import build_clip_index
from bird_cv.classification.extract_behavior_clips import extract_behavior_clips
from bird_cv.pipelines.config import load_config, resolve_run_dir
from bird_cv.preprocessing.annotations_to_yolo import stream_annotations_to_yolo
from bird_cv.preprocessing.crop_yolo_labels import run_crop_yolo
from bird_cv.preprocessing.get_label_tables import get_label_tables
from bird_cv.preprocessing.get_split_guidance import split_camera_data
from bird_cv.segmentation.frames import extract_all_frames
from bird_cv.segmentation.segment import segment
from bird_cv.utils import extract_camera_video
import tempfile


class SplitRatio(msgspec.Struct):
    train: float
    val: float
    test: float


class Paths(msgspec.Struct):
    videos_path: str
    segmentation_configs_path: str
    sam2_checkpoint_path: str
    label_paths: list[str]


class Preprocessing(msgspec.Struct):
    num_frames: int = 16
    stride: int = 8
    random_seed: int = 42


class Run(msgspec.Struct):
    base_path: str
    run_id: str = ""


class PreprocessingConfig(msgspec.Struct):
    run: Run
    paths: Paths
    preprocessing: Preprocessing
    split_ratio: SplitRatio


def run_preprocessing_pipeline(config_path: Path) -> None:
    """Run the full preprocessing pipeline from label annotations to video clips.

    Steps:
        1. Convert Label Studio JSON exports to label tables
        2. Generate split guidance (train/val/test)
        3. Extract YOLO images and labels from videos
        4. Run SAM2 segmentation on target frames
        5. Crop YOLO images and labels per cage
        6. Extract fixed-length behavior clips for VideoMAE

    Args:
        config_path: Path to the pipeline TOML config file.
    """
    cfg = load_config(config_path, PreprocessingConfig)

    run_dir = resolve_run_dir(Path(cfg.run.base_path), cfg.run.run_id or None)
    intermediate_path = run_dir / "intermediate"
    intermediate_path.mkdir(exist_ok=True, parents=True)
    yolo_store_path = run_dir / "yolo_store"
    yolo_store_path.mkdir(exist_ok=True, parents=True)

    videos_path = Path(cfg.paths.videos_path)
    label_paths = [Path(p) for p in cfg.paths.label_paths]
    split_ratio = {
        "train": cfg.split_ratio.train,
        "val": cfg.split_ratio.val,
        "test": cfg.split_ratio.test,
    }

    # Step 1: Build label tables from each labeler
    print("\n=== Step 1: Building label tables ===")
    video_store, frame_store = [], []
    for ii, label_path in enumerate(label_paths):
        get_label_tables(
            label_json_path=label_path / "annotations.json",
            videos_path=videos_path,
            output_dir=intermediate_path / f"input_{ii}",
            num_frames=cfg.preprocessing.num_frames,
        )
        video_store.append(
            pl.read_ndjson(intermediate_path / f"input_{ii}" / "video_data.ndjson")
        )
        frame_store.append(
            pl.read_ndjson(intermediate_path / f"input_{ii}" / "frame_data.ndjson")
        )

    pl.concat(video_store).write_ndjson(intermediate_path / "video_data_full.ndjson")
    pl.concat(frame_store).write_ndjson(intermediate_path / "frame_data_full.ndjson")

    # Step 2: Generate split guidance
    print("\n=== Step 2: Generating split guidance ===")
    split_camera_data(
        video_data_path=intermediate_path / "video_data_full.ndjson",
        frame_data_path=intermediate_path / "frame_data_full.ndjson",
        split_ratio=split_ratio,
        output_path=intermediate_path,
        random_seed=cfg.preprocessing.random_seed,
    )

    # Step 3: Extract YOLO images and labels
    print("\n=== Step 3: Extracting YOLO images and labels ===")
    for label_path in label_paths:
        stream_annotations_to_yolo(
            path_to_videos=videos_path,
            path_to_annotations=label_path / "annotations_interpolated.json",
            path_to_output=yolo_store_path,
            path_to_guidance=intermediate_path / "split_guidance.parquet",
            processes=1,
        )

    # Step 4: Run SAM2 segmentation
    print("\n=== Step 4: Running SAM2 segmentation ===")
    segmentation_configs_path = Path(cfg.paths.segmentation_configs_path)
    model_checkpoint_path = Path(cfg.paths.sam2_checkpoint_path)
    split_guidance = pl.read_parquet(intermediate_path / "split_guidance.parquet")
    segmentations_path = run_dir / "segmentations"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for video_str, target_frames in split_guidance.select(
            "video_path", "target_frames"
        ).to_numpy():
            camera_id, video_name = extract_camera_video(video_str=video_str)
            video_id = Path(video_name).stem
            print(f"  Segmenting {camera_id}/{video_id}")

            frame_store_path = temp_path / camera_id / video_id
            frame_store_path.mkdir(exist_ok=True, parents=True)
            extract_all_frames(
                video_path=videos_path / camera_id / video_name,
                output_path=frame_store_path,
                frames=target_frames,
            )

            cam_pred_path = (
                segmentations_path / camera_id / video_id / "segmentation.json"
            )
            if not cam_pred_path.exists():
                train_video_id = next(
                    iter((segmentation_configs_path / "frames" / camera_id).glob("*/"))
                ).name
                segment(
                    config_path=segmentation_configs_path
                    / "configs"
                    / f"{camera_id}.json",
                    x0_frame_path=segmentation_configs_path
                    / "frames"
                    / camera_id
                    / train_video_id
                    / "00001.jpg",
                    y_video_path=frame_store_path,
                    model_checkpoint_path=model_checkpoint_path,
                    output_path=cam_pred_path,
                    device="cuda",
                    visualize=False,
                )

    # Step 5: Build clip index and crop YOLO per cage
    print("\n=== Step 5: Cropping YOLO images per cage ===")
    build_clip_index(
        frame_data_path=intermediate_path / "frame_data_full.ndjson",
        video_data_path=intermediate_path / "video_data_full.ndjson",
        split_guidance_path=intermediate_path / "split_guidance.parquet",
        output_path=intermediate_path / "clip_index.parquet",
    )
    run_crop_yolo(
        yolo_data_path=yolo_store_path,
        yolo_output_path=run_dir / "yolo_crop",
        video_segments_path=segmentations_path,
        clip_index_path=intermediate_path / "clip_index.parquet",
        clip_output_path=intermediate_path / "behavior_index.parquet",
    )

    # Step 6: Extract behavior clips
    print("\n=== Step 6: Extracting behavior clips ===")
    extract_behavior_clips(
        behavior_index_path=intermediate_path / "behavior_index.parquet",
        cropped_frames_path=run_dir / "yolo_crop",
        output_path=run_dir / "video_crop",
        num_frames=cfg.preprocessing.num_frames,
        stride=cfg.preprocessing.stride,
    )

    print(f"\nPreprocessing complete. Outputs at: {run_dir}")


if __name__ == "__main__":
    run_preprocessing_pipeline(config_path=Path(__file__).parent / "config.toml")
