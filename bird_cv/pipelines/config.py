"""Shared config loading utilities for all pipelines."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from datetime import datetime
import os
import msgspec
import tomllib

time = datetime.now().strftime("%Y%m%d_%H%M%S")

API_KEY_ENV_VAR = "LABEL_STUDIO_API_KEY"


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
    # Paths auto-resolved by BirdCVConfig.__post_init__ (not in TOML). These
    # need a placeholder default, not just init=False, so msgspec.convert
    # doesn't treat them as required TOML fields.
    path_to_training_data: Path = field(init=False, default=Path())
    path_to_output: Path = field(init=False, default=Path())


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
    # Paths auto-resolved by BirdCVConfig.__post_init__ (not in TOML). These
    # need a placeholder default, not just init=False, so msgspec.convert
    # doesn't treat them as required TOML fields.
    path_to_training_data: Path = field(init=False, default=Path())
    path_to_output: Path = field(init=False, default=Path())


@dataclass
class PrepareConfig:
    """Configuration for snapshotting, splitting and preparing training data.

    Optional ``[prepare]`` TOML table; every field has a default so configs
    written before it existed still load.
    """

    split_ratio: dict[str, float] = field(
        default_factory=lambda: {"train": 0.70, "val": 0.15, "test": 0.15}
    )
    split_seed: int = 42
    cv_folds: int = 3
    # Every k-th frame for Detect videos that have no Classify labels yet.
    yolo_uniform_stride: int = 5
    # Frames between consecutive VideoMAE clip starts within a labeled stretch.
    videomae_clip_stride: int = 8


@dataclass
class InferConfig:
    """Configuration for end-to-end inference. Optional ``[infer]`` TOML table."""

    chunk_frames: int = 900
    # Frames of overlap between consecutive VideoMAE chunks.
    chunk_overlap: int = 16
    # Carry the last box forward through gaps of at most this many frames.
    gap_fill_frames: int = 8
    # Detection confidence threshold for e2e output (not Label Studio pre-labels).
    det_conf: float = 0.25
    crop_size: int = 224


@dataclass
class EvaluateConfig:
    """Configuration for evaluation. Optional ``[evaluate]`` TOML table."""

    iou_threshold: float = 0.5
    sweep_gap_fill_frames: list[int] = field(default_factory=lambda: [0, 8, 16])
    sweep_det_conf: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.5])
    # Paths below are relative to path_to_base_output unless absolute.
    # Ground-truth parquet (GROUND_TRUTH_SCHEMA), or a directory of them.
    ground_truth: Path = Path("ground_truth")
    # Directory of split_v{n}.parquet (SPLIT_SCHEMA); the highest n is used.
    splits: Path = Path("splits")
    # CV fold assignment (CV_FOLD_SCHEMA).
    cv_folds: Path = Path("splits/cv_folds.parquet")
    # Prediction runs (PREDICTION_SCHEMA): {val_sweep/gap{g}_conf{c}, test,
    # cv/fold{k}}/ under this directory.
    predictions: Path = Path("predictions")
    output: Path = Path("evaluation")
    # Non-test cameras held out as val within each CV fold.
    cv_val_cameras: int = 1


@dataclass
class ModelSpec:
    """Which model fills a pipeline role and how to train it.

    ``kind`` picks the trainer in ``bird_cv.training.registry``; nothing
    else in the pipelines names a model. A ``[train.detector]`` or
    ``[train.classifier]`` table overrides only the keys it sets, unless it
    changes ``kind``: then it starts from a bare spec (these field
    defaults) and must set ``pretrained`` and ``dataset``.

    Attributes:
        kind: Registry key, e.g. ``"yolo"`` or ``"videomae"``.
        pretrained: Generic pretrained weights to start from (ADR 0004).
        dataset: Training data root, relative to ``path_to_base_output``
            unless absolute; CV folds read ``{dataset}/fold{k}``.
        epochs: Epochs per training run or tuning trial.
        tune: Tune hyperparameters on the frozen split before training.
        tune_trials: Number of tuning trials.
        search_space: Parameter name to distribution spec, e.g.
            ``{type = "loguniform", lower = 1e-5, upper = 1e-3}`` or
            ``{type = "choice", values = [true, false]}`` (types:
            ``uniform``, ``loguniform``, ``randint`` with ``lower`` and
            ``upper``; ``choice`` with ``values``). Names are the trainer's
            training arguments. Empty means the trainer's own default
            space, if it has one.
        options: Training arguments for every run of this role, tuned or
            not (e.g. ``batch_size``).
        tune_options: Arguments for the tuner only (e.g. ``grace_period``,
            ``cpus_per_trial``).
    """

    kind: str
    pretrained: str
    dataset: Path
    epochs: int = 10
    tune: bool = True
    tune_trials: int = 20
    search_space: dict[str, dict] = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    tune_options: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Training from generic pretrained weights (ADR 0004). Optional ``[train]`` table.

    Holds one :class:`ModelSpec` per role, as the optional tables
    ``[train.detector]`` and ``[train.classifier]``.
    """

    # Relative to path_to_base_output unless absolute; runs go to
    # {output}/{role}/{run_id}.
    output: Path = Path("train")
    device: int = 0
    detector: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            kind="yolo",
            pretrained="yolo11n.pt",
            # images/{train,val}, labels/{train,val}
            dataset=Path("datasets/yolo"),
            epochs=60,
            tune_trials=30,
            # Empty: Ultralytics' default Ray Tune space.
            search_space={},
        )
    )
    classifier: ModelSpec = field(
        default_factory=lambda: ModelSpec(
            kind="videomae",
            pretrained="MCG-NJU/videomae-base",
            # {train,val}/{label}/{clip}/
            dataset=Path("datasets/videomae_clips"),
            epochs=10,
            tune_trials=20,
            search_space={
                "lr": {"type": "loguniform", "lower": 1e-5, "upper": 1e-3},
                "freeze_encoder": {"type": "choice", "values": [True, False]},
                "weight_decay": {"type": "loguniform", "lower": 1e-4, "upper": 1e-1},
                "class_weighted_loss": {"type": "choice", "values": [True, False]},
            },
            options={"num_frames": 16, "batch_size": 8},
            tune_options={"cpus_per_trial": 8, "grace_period": 2},
        )
    )

    def roles(self) -> dict[str, ModelSpec]:
        """Every role (a :class:`ModelSpec` field), by name."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), ModelSpec)
        }


def _convert_train_config(table: dict, dec_hook) -> TrainConfig:
    """Build :class:`TrainConfig`, merging role tables onto their defaults.

    A role table that keeps the default ``kind`` overrides only the keys it
    sets; one that changes ``kind`` starts from a bare :class:`ModelSpec`.
    Kinds and search spaces are validated against
    :mod:`bird_cv.training.registry` (without importing Ray).

    Raises:
        ValueError: On unknown keys in ``[train]`` or a role table, an
            unknown ``kind``, a new ``kind`` without ``pretrained`` and
            ``dataset``, or a malformed ``search_space`` entry.
    """
    from bird_cv.training.registry import check_search_space, trainer_kinds

    defaults = TrainConfig().roles()
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(table) - known
    if unknown:
        raise ValueError(f"Unknown keys in [train]: {sorted(unknown)}")
    spec_keys = {f.name for f in fields(ModelSpec)}
    merged = dict(table)
    for role, default in defaults.items():
        where = f"[train.{role}]"
        role_table = table.get(role, {})
        unknown = set(role_table) - spec_keys
        if unknown:
            raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}")
        kind = role_table.get("kind", default.kind)
        if kind not in trainer_kinds():
            raise ValueError(
                f"{where} kind {kind!r} has no trainer; known: {trainer_kinds()}"
            )
        if kind == default.kind:
            base = msgspec.to_builtins(default, enc_hook=lambda p: str(p))
        else:
            missing = sorted({"pretrained", "dataset"} - set(role_table))
            if missing:
                raise ValueError(
                    f"{where} changes kind to {kind!r}, so it must set {missing}"
                )
            base = {}
        merged[role] = base | role_table
        check_search_space(merged[role].get("search_space", {}), where=where)
    return msgspec.convert(merged, type=TrainConfig, dec_hook=dec_hook)


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
    # Optional sub-configs for the new pipelines
    prepare: PrepareConfig = field(default_factory=PrepareConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
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


def load_config(
    config_path: Path,
    config_type: type = BirdCVConfig,
    *,
    require_api_key: bool = True,
):
    """Load and decode a TOML config file into a typed config object.

    BirdCVConfig.detect and .classify are required nested dataclasses
    (DetectConfig / ClassifyConfig) built from the [detect] / [classify]
    TOML tables. Since those dataclasses have no valid zero-arg
    constructor, they're decoded separately here and passed in
    explicitly rather than relying on a default_factory.

    Args:
        config_path: Path to the TOML file.
        config_type: The top-level config class to build (BirdCVConfig).
        require_api_key: If False (pipelines that never talk to Label
            Studio, e.g. inference), a missing API key is allowed and left
            empty instead of raising.

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
    # Optional tables for the new pipelines; absent means all defaults.
    # Unknown keys raise, so a typo can't silently fall back to a default.
    optional_cfgs = {}
    for name, typ in (
        ("prepare", PrepareConfig),
        ("infer", InferConfig),
        ("evaluate", EvaluateConfig),
    ):
        table = raw.pop(name, {})
        unknown = set(table) - {f.name for f in fields(typ)}
        if unknown:
            raise ValueError(f"Unknown keys in [{name}]: {sorted(unknown)}")
        optional_cfgs[name] = msgspec.convert(table, type=typ, dec_hook=str_to_path)
    # [train] nests per-role tables merged onto role defaults (same check).
    optional_cfgs["train"] = _convert_train_config(
        raw.pop("train", {}), dec_hook=str_to_path
    )

    # The Label Studio key comes from the environment when set, so it never
    # has to live in a committed TOML; a TOML value still works as a fallback.
    env_api_key = os.environ.get(API_KEY_ENV_VAR)
    if env_api_key:
        raw["api_key"] = env_api_key
    elif not raw.get("api_key") and not require_api_key:
        raw["api_key"] = ""
    elif not raw.get("api_key"):
        raise ValueError(
            f"No Label Studio API key: set {API_KEY_ENV_VAR} or api_key in {config_path}"
        )

    top_level_types = {f.name: f.type for f in fields(config_type)}
    raw = {
        key: str_to_path(top_level_types.get(key), value) for key, value in raw.items()
    }

    return config_type(**raw, detect=detect_cfg, classify=classify_cfg, **optional_cfgs)


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
