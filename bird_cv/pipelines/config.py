"""Shared config loading utilities for all pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import msgspec
import tomllib

time = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class DetectConfig:
    """Configuration for the detection pipeline."""

    # Paths that must be filled in TOML
    path_to_yolo: Path
    path_to_yolo_config: Path
    path_to_tracker_config: Path
    # Training parameters
    epochs: int = 30
    device: int = 0
    tune: bool = False
    tune_iterations: int = 30
    run_name: str = "bird_yolo"
    # Paths auto-resolved by BirdCVConfig (not in TOML)
    path_to_training_data: Path = field(init=False)
    path_to_output: Path = field(init=False)


@dataclass
class ClassifyConfig:
    """Configuration for the behavior classification pipeline."""

    # Paths that must be filled in TOML
    path_to_mae: Path
    # Training parameters
    num_frames: int = 16
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    device: str = "cuda"
    freeze_encoder: bool = True
    # Paths auto-resolved by BirdCVConfig (not in TOML)
    path_to_training_data: Path = field(init=False)
    path_to_output: Path = field(init=False)


@dataclass
class BirdCVConfig:
    # Core paths
    path_to_base_output: Path
    path_to_raw_videos: Path
    path_to_segmentation_configs: Path
    # Hugging Face Hub model id for the SAM2 checkpoint (e.g.
    # "facebook/sam2.1-hiera-large"), downloaded automatically on first use.
    sam_model_id: str
    # Label Studio
    api_key: str
    detect_project_name: str
    classify_project_name: str
    # Sub-configs (filled from TOML then resolved here)
    detect: DetectConfig
    classify: ClassifyConfig
    # Derived paths
    path_to_cage_videos: Path = field(init=False)
    path_to_bbox_videos: Path = field(init=False)
    path_to_cropped_cage_frames: Path = field(init=False)
    path_to_cropped_bbox_frames: Path = field(init=False)
    path_to_detect_annotations: Path = field(init=False)
    path_to_classify_annotations: Path = field(init=False)
    path_to_guidance: Path = field(init=False)
    path_to_guidance_within_camera: Path = field(init=False)
    path_to_segmentation_output: Path = field(init=False)

    def __post_init__(self):
        # Videos
        self.path_to_cage_videos = self.path_to_base_output / "cage_videos"
        self.path_to_bbox_videos = self.path_to_base_output / "cropped_bbox_videos"
        # Frames
        self.path_to_cropped_cage_frames = (
            self.path_to_base_output / "cropped_cage_frames"
        )
        self.path_to_cropped_bbox_frames = (
            self.path_to_base_output / "cropped_bbox_frames"
        )
        # Annotations
        self.path_to_detect_annotations = (
            self.path_to_base_output / "annotations" / f"{time}_detect_annotations.json"
        )
        self.path_to_classify_annotations = (
            self.path_to_base_output
            / "annotations"
            / f"{time}_classify_annotations.json"
        )
        # Split guidance
        self.path_to_guidance = (
            self.path_to_base_output / "split_guidance" / "mock_split_guidance.parquet"
        )
        self.path_to_guidance_within_camera = (
            self.path_to_base_output
            / "split_guidance"
            / "mock_split_guidance_within_camera.parquet"
        )
        # Segmentation
        self.path_to_segmentation_output = (
            self.path_to_base_output / "segmentations" / "segmentations.json"
        )
        # Resolve sub-config paths
        self.detect.path_to_training_data = (
            self.path_to_base_output / "detect" / "training_data"
        )
        self.detect.path_to_output = self.path_to_base_output / "detect" / "output"
        self.classify.path_to_training_data = (
            self.path_to_base_output / "classify" / "training_data"
        )
        self.classify.path_to_output = self.path_to_base_output / "classify" / "output"


def load_config(config_path: Path, config_type: type = BirdCVConfig):
    """Load and decode a TOML config file into a typed config object.

    BirdCVConfig.detect and .classify are required nested dataclasses
    (DetectConfig / ClassifyConfig) built from the [detect] / [classify]
    TOML tables. Since those dataclasses have no valid zero-arg
    constructor, they're decoded separately here and passed in
    explicitly rather than relying on a default_factory.

    Args:
        config_path: Path to the TOML file.
        config_type: The top-level config class to build (BirdCVConfig).

    Returns:
        A fully constructed config_type instance.
    """
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    detect_data = raw.pop("detect")
    classify_data = raw.pop("classify")

    def str_to_path(typ, obj):
        if typ is Path and isinstance(obj, str):
            return Path(obj)
        return obj

    detect_cfg = msgspec.convert(detect_data, type=DetectConfig, dec_hook=str_to_path)
    classify_cfg = msgspec.convert(
        classify_data, type=ClassifyConfig, dec_hook=str_to_path
    )

    return config_type(**raw, detect=detect_cfg, classify=classify_cfg)


def save_config(config: msgspec.Struct, config_path: Path) -> None:
    """Save a msgspec struct to a TOML config file.
    Args:
        config: A msgspec.Struct instance to encode.
        config_path: Path to save the TOML file.
    """

    def path_to_str(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Unsupported type: {type(obj)}")

    config_path.write_bytes(msgspec.toml.encode(config, enc_hook=path_to_str))


def resolve_run_dir(base_path: Path, run_id: str | None) -> Path:
    """Resolve the output directory for a pipeline run.

    If ``run_id`` is None or empty, generates a timestamped ID of the form
    ``YYYYMMDD_HHMMSS``. Otherwise uses the provided value.

    Args:
        base_path: Root directory under which the run directory is created.
        run_id: Optional fixed run identifier.

    Returns:
        Path to the run directory (not yet created).
    """
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / run_id
    print(f"Run directory: {run_dir}")
    run_dir.mkdir(exist_ok=True, parents=True)
    return run_dir
