import numpy as np
import polars as pl
from pathlib import Path


def split_camera_data(
    video_data_path: Path,
    frame_data_path: Path,
    output_path: Path,
    split_ratio: dict[str, float],
    random_seed: int = 42,
) -> None:
    """Splits video and frame data by camera into train, validation, and test sets and saves to disk.

    This function reads video and frame metadata, randomly splits cameras into train/val/test
    based on the specified ratios, subsamples frames, optionally adds resting frames, and
    writes the resulting lookup tables to parquet files.

    Args:
        video_data_path (Path): Path to the NDJSON file containing video-level metadata.
        frame_data_path (Path): Path to the NDJSON file containing frame-level metadata.
        output_path (Path): Directory where the split lookup parquet files will be saved.
        split_ratio (dict[str, float]): Dictionary specifying the train/val/test split ratios.
            Keys must include "train", "val", and "test".
        random_seed (int, optional): Seed for random number generation to ensure reproducibility.
            Defaults to 42.

    Returns:
        None. Parquet files for each split ("train_lookup.parquet", "val_lookup.parquet",
        "test_lookup.parquet") are written to `output_path`.
    """

    # Set random seed
    np.random.seed(random_seed)

    # Load in summary tables
    video_data = pl.read_ndjson(video_data_path)
    frame_data = pl.read_ndjson(frame_data_path)

    # Randomly split the cameras into train / validation / test
    cameras = video_data.select("camera_id").unique()
    n_cameras = cameras.height
    indices = np.random.permutation(n_cameras)
    n_train = round(split_ratio["train"] * n_cameras)
    n_val = round(split_ratio["val"] * n_cameras)
    train_idx, val_idx, test_idx = np.split(indices, [n_train, n_train + n_val])

    split_cameras = {}
    split_cameras["train"] = cameras[train_idx].get_column("camera_id")
    split_cameras["val"] = cameras[val_idx].get_column("camera_id")

    # Move the train and val data for yolo
    for split, cams in split_cameras.items():
        # Get relevent videos and frames for each camera
        split_videos = video_data.filter(pl.col("camera_id").is_in(cams.implode()))
        split_frames = frame_data.join(split_videos, on="video_id", how="inner")

        # Subsample the frames of each video
        split_frames = subsample_frames(split_frames)

        # Provide additional resting frames based on max frame count of other actions
        split_frames = sample_resting_frames(split_frames, seed=random_seed)

        videos = split_videos.with_columns(
            (pl.col("framesCount") / pl.col("duration")).alias("fps")
        )
        videos = videos.select("video_id", "video_path", "fps")

        # Get the video path
        split_frames = split_frames.join(videos, on="video_id")

        # Save split frames
        split_frames.write_parquet(output_path / f"{split}_lookup.parquet")


def subsample_frames(frame_data: pl.DataFrame) -> pl.DataFrame:
    """Subsamples non-resting frames and computes target frames and label counts per video.

    This function extracts all frames corresponding to non-resting labels, generates a
    list of target frames per video, and calculates counts for each label. It returns a
    Polars DataFrame with target frames and label counts, merged with the total frame
    count per video.

    Args:
        frame_data (pl.DataFrame): Polars DataFrame containing frame-level annotations. Must
            include columns "video_id", "frame_begin", "frame_end", "label", and "framesCount".

    Returns:
        pl.DataFrame: DataFrame with the following columns:
            - video_id: Unique identifier for each video.
            - target_frames: List of frame numbers corresponding to non-resting labels.
            - label_counts: Struct containing counts of each label per video.
            - framesCount: Total number of frames in the video.
    """

    # For each video, I want a list of all relevant frames as well as count of each behavior
    non_resting_frames = frame_data.filter(pl.col("label") != "Resting")

    # Group by video_id, create ranges, and compute unique sorted values per group
    # Create row-wise ranges
    df = non_resting_frames.with_columns(
        pl.int_ranges(pl.col("frame_begin"), pl.col("frame_end"), 1).alias("range")
    )

    # Explode ranges
    df_exploded = df.explode("range")

    # Compute target frames per video_id
    unique_ranges = df_exploded.group_by("video_id").agg(
        [pl.col("range").unique().sort().alias("target_frames")]
    )

    # Compute label counts per video_id
    label_counts = (
        df_exploded.group_by(["video_id", "label"])
        .agg(pl.len().alias("count"))
        .pivot(values="count", index="video_id", on="label")
        .fill_null(0)
        .with_columns(
            [pl.struct([c for c in df["label"].unique()]).alias("label_counts")]
        )
        .select(["video_id", "label_counts"])
    )

    # Join both together
    result = unique_ranges.join(label_counts, on="video_id")

    # Sample a portion of random frames that have resting (or no label)
    result = result.join(
        non_resting_frames.select("video_id", "framesCount").unique(),
        on="video_id",
        how="left",
    )

    return result


def sample_resting_frames(frame_target_df: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Samples additional resting frames to balance frame counts across labels.

    Given a DataFrame with target frames and label counts, this function samples frames
    not already included in `target_frames` (typically "Resting" frames) so that each
    video has enough frames to match the maximum label count. The resulting target frames
    are concatenated with the sampled resting frames.

    Args:
        frame_target_df (pl.DataFrame): DataFrame containing columns:
            - target_frames: List of preselected frame numbers for each video.
            - label_counts: Struct with counts of each label for the video.
            - framesCount: Total number of frames in the video.
        seed (int): Random seed for reproducible sampling of resting frames.

    Returns:
        pl.DataFrame: Updated DataFrame with the same columns as input, but `target_frames`
        now includes the sampled additional resting frames. Temporary columns used for
        computation are dropped.
    """

    # Get the names of the fields within the struct
    label_fields = frame_target_df.schema["label_counts"].fields
    label_names = [field.name for field in label_fields]

    result = (
        frame_target_df.unnest("label_counts")
        .with_columns((pl.max_horizontal(label_names)).alias("max_count"))
        .with_columns(
            pl.int_ranges(1, pl.col("framesCount") + 1, 1).alias("all_frames")
        )
        .with_columns(
            pl.col("all_frames")
            .list.set_difference("target_frames")
            .alias("candidate_frames")
        )
        .with_columns(
            pl.when(
                pl.col("max_count")
                > (pl.col("framesCount") - pl.sum_horizontal(label_names))
            )
            .then(
                pl.col("candidate_frames")
                .list.sample(
                    n=pl.col("framesCount") - pl.sum_horizontal(label_names), seed=seed
                )
                .alias("sampled_additional_frames"),
            )
            .otherwise(
                pl.col("candidate_frames")
                .list.sample(n=pl.col("max_count"), seed=seed)
                .alias("sampled_additional_frames"),
            ),
            pl.when(
                pl.col("max_count")
                > (pl.col("framesCount") - pl.sum_horizontal(label_names))
            )
            .then(
                (pl.col("framesCount") - pl.sum_horizontal(label_names)).alias(
                    "added_rests"
                )
            )
            .otherwise(pl.col("max_count").alias("added_rests")),
        )
    )

    # Finally concat all targeted frames
    result = result.with_columns(
        pl.col("target_frames")
        .list.concat("sampled_additional_frames")
        .alias("target_frames")
    )

    return result.drop(
        "max_count",
        "all_frames",
        "candidate_frames",
        "sampled_additional_frames",
        "framesCount",
    )
