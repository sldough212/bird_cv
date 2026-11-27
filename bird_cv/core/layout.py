"""Every output path under ``path_to_base_output``, defined once.

All new pipeline stages (prepare, dataset build, train, infer, evaluate) and
the slurm scripts locate their inputs and outputs through :class:`Layout`, so
a path changes here or nowhere. One pipeline run uses one id (``RUN_ID`` in
``submit.sh``) for every stage::

    splits/split_v{n}.parquet                 SPLIT_SCHEMA, shared by every dataset
    splits/split_v{n}/manifest.json
    datasets/{dataset_id}/                    prepare + build_datasets
        manifest.json                         its key: snapshot, ground truth,
                                              split and sampling hashes
        annotations/detect.json, classify.json   Label Studio exports
        ground_truth.parquet, ground_truth_stats.json
        split.parquet                         the split version it used
        split_summary.parquet, yolo_frames.parquet, videomae_clips.parquet
        {role}/{kind}/                        training data, with selection,
                                              split, split_guidance (.parquet)
                                              and manifest.json
    models/{role}/{run_id}/                   a training run (TrainRun)
    infer/{run_id}/                           an ad-hoc inference run, or the root of:
        val_sweep/gap{g}_conf{c}/             one val-sweep setting
        test/                                 test cameras at the chosen setting
    eval/{run_id}/{mode or role}/             sweep, test, or a role's model metrics
    slurm_logs/{run_id}/                      submit.sh job logs

A dataset whose key matches an existing one is a symlink to it (see
``bird_cv/pipelines/prepare.py``). Files inside run directories are named by
:class:`TrainRun`, :class:`InferRun` and :class:`EvalRun`. The legacy Label
Studio pipelines keep their own paths (``BirdCVConfig.__post_init__``).

TODO: inference standardizes, segments (SAM2) and detects each video again
in every run and every val-sweep setting, though those steps depend only on
the raw video and the segmenter/detector. Cache them (e.g.
``cache/standardized/`` and ``cache/segments/{segmenter}/``) so sweep
settings only redo gap fill and classification.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def default_run_id(run_id: str | None) -> str:
    """``run_id``, or a ``YYYYMMDD_HHMMSS`` timestamp if it is None or empty."""
    return run_id or datetime.now().strftime("%Y%m%d_%H%M%S")


def split_file(splits_dir: Path, version: int) -> Path:
    """Split version ``version`` inside a splits directory."""
    return splits_dir / f"split_v{version}.parquet"


def split_version_of(path: Path) -> int | None:
    """The version of a :func:`split_file` path, or None for any other file."""
    m = re.fullmatch(r"split_v(\d+)\.parquet", path.name)
    return int(m.group(1)) if m else None


def split_manifest_dir(splits_dir: Path, version: int) -> Path:
    """Directory holding split version ``version``'s ``manifest.json``."""
    return splits_dir / f"split_v{version}"


def snapshot_files(directory: Path) -> tuple[Path, Path]:
    """The Detect and Classify exports in a snapshot or export directory."""
    return directory / "detect.json", directory / "classify.json"


def sweep_setting_name(gap_fill_frames: int, det_conf: float) -> str:
    """Directory name of one val-sweep setting, e.g. ``gap8_conf0.25``."""
    return f"gap{gap_fill_frames}_conf{det_conf:g}"


@dataclass(frozen=True)
class TrainRun:
    """Files of one training run directory.

    Attributes:
        dir: The run directory.
    """

    dir: Path

    @property
    def best_hyperparameters(self) -> Path:
        """Selected training arguments, reusable with ``--hyperparameters``."""
        return self.dir / "best_hyperparameters.json"

    @property
    def train_result(self) -> Path:
        """Checkpoint and validation metrics of the selected model."""
        return self.dir / "train_result.json"

    @property
    def selection(self) -> Path:
        """Every candidate checkpoint and its selection score."""
        return self.dir / "selection.parquet"

    @property
    def tune_dir(self) -> Path:
        """Where a tuner keeps its trials (itself laid out as a :class:`TrainRun`)."""
        return self.dir / "tune"

    @property
    def trials(self) -> Path:
        """One row per tuning trial."""
        return self.dir / "trials.csv"


@dataclass(frozen=True)
class EvalRun:
    """Files of one evaluation output directory.

    Attributes:
        dir: The output directory.
    """

    dir: Path

    @property
    def metrics(self) -> Path:
        """Headline end-to-end scores."""
        return self.dir / "metrics.json"

    @property
    def per_class(self) -> Path:
        return self.dir / "per_class.parquet"

    @property
    def confusion(self) -> Path:
        return self.dir / "confusion.parquet"

    @property
    def per_camera(self) -> Path:
        return self.dir / "per_camera.parquet"

    @property
    def mot(self) -> Path:
        return self.dir / "mot.parquet"

    @property
    def sweep(self) -> Path:
        """One row per val-sweep setting, best first."""
        return self.dir / "sweep.parquet"

    @property
    def best_setting(self) -> Path:
        """The (gap fill, det_conf) setting chosen on val."""
        return self.dir / "best_setting.json"

    @property
    def model_metrics(self) -> Path:
        """One role's model-level scores on val and test."""
        return self.dir / "model_metrics.json"

    def model_split(self, split: str) -> Path:
        """Where a trainer writes its model-level outputs for one split."""
        return self.dir / split


@dataclass(frozen=True)
class InferRun:
    """Files of one inference run directory, shared by all its array tasks.

    Attributes:
        dir: The run directory.
    """

    dir: Path

    def standardized(self, camera_id: str, video_id: str) -> Path:
        """The standardized (15 fps) copy of a raw video."""
        return self.dir / "standardized" / camera_id / f"{video_id}.mp4"

    def detections(self, camera_id: str, video_id: str) -> Path:
        """Raw tracker output for a video (``DETECTION_SCHEMA``)."""
        return self.dir / "detections" / camera_id / f"{video_id}.parquet"

    @property
    def predictions_dir(self) -> Path:
        """Root of every video's predictions (``PREDICTION_SCHEMA``)."""
        return self.dir / "predictions"

    def predictions(self, camera_id: str, video_id: str) -> Path:
        """One video's predictions."""
        return self.predictions_dir / camera_id / f"{video_id}.parquet"

    def timings(self, task_index: int, num_tasks: int) -> Path:
        """Stage timings of one task (``TIMINGS_SCHEMA``)."""
        if num_tasks == 1:
            return self.dir / "timings.parquet"
        return self.dir / f"timings_task{task_index:03d}.parquet"

    def task_manifest(self, task_index: int) -> Path:
        """One task's manifest, so array tasks never overwrite each other."""
        return self.dir / f"manifest_task{task_index:03d}.json"

    def task_manifests(self) -> list[Path]:
        """Every task's manifest, sorted."""
        return sorted(self.dir.glob("manifest_task*.json"))

    def failed(self, task_index: int) -> Path:
        """One task's failed videos and their errors."""
        return self.dir / f"failed_task{task_index:03d}.json"


@dataclass(frozen=True)
class Layout:
    """Paths under one ``path_to_base_output``.

    Attributes:
        base: ``path_to_base_output``.
    """

    base: Path

    # Splits
    @property
    def splits(self) -> Path:
        """Directory of every split version."""
        return self.base / "splits"

    def split(self, version: int) -> Path:
        """Split version ``version`` (``SPLIT_SCHEMA``)."""
        return split_file(self.splits, version)

    # Datasets
    @property
    def datasets(self) -> Path:
        """Root of every dataset, ``{dataset_id}/`` below it."""
        return self.base / "datasets"

    def dataset_dir(self, dataset_id: str) -> Path:
        """One dataset: prepare's outputs and each role's training data."""
        return self.datasets / dataset_id

    def annotations(self, dataset_id: str) -> Path:
        """The dataset's Label Studio exports (see :func:`snapshot_files`)."""
        return self.dataset_dir(dataset_id) / "annotations"

    def ground_truth(self, dataset_id: str) -> Path:
        """Ground truth table (``GROUND_TRUTH_SCHEMA``)."""
        return self.dataset_dir(dataset_id) / "ground_truth.parquet"

    def ground_truth_stats(self, dataset_id: str) -> Path:
        """Counts of what the ground-truth conversion fixed or dropped."""
        return self.dataset_dir(dataset_id) / "ground_truth_stats.json"

    def dataset_split_file(self, dataset_id: str) -> Path:
        """Copy of the split version the dataset was sampled on."""
        return self.dataset_dir(dataset_id) / "split.parquet"

    def split_summary(self, dataset_id: str) -> Path:
        """Per-split class counts (``SPLIT_SUMMARY_SCHEMA``)."""
        return self.dataset_dir(dataset_id) / "split_summary.parquet"

    def yolo_frames(self, dataset_id: str) -> Path:
        """Sampled detector training frames (``YOLO_FRAMES_SCHEMA``)."""
        return self.dataset_dir(dataset_id) / "yolo_frames.parquet"

    def videomae_clips(self, dataset_id: str) -> Path:
        """Sampled classifier training clips (``VIDEOMAE_CLIPS_SCHEMA``)."""
        return self.dataset_dir(dataset_id) / "videomae_clips.parquet"

    def dataset(self, dataset_id: str, role: str, kind: str) -> Path:
        """Training data of one role's model kind."""
        return self.dataset_dir(dataset_id) / role / kind

    def dataset_split(self, dataset_id: str, role: str, kind: str) -> Path:
        """The camera split a dataset was built from (``camera_id``, ``split``)."""
        return self.dataset(dataset_id, role, kind) / "split.parquet"

    def dataset_selection(self, dataset_id: str, role: str, kind: str) -> Path:
        """The sampled frames or clips a dataset was built from."""
        return self.dataset(dataset_id, role, kind) / "selection.parquet"

    def dataset_guidance(self, dataset_id: str, role: str, kind: str) -> Path:
        """The split-guidance parquet handed to the dataset converter."""
        return self.dataset(dataset_id, role, kind) / "split_guidance.parquet"

    def resolve_dataset(self, dataset_id: str | None, run_id: str | None) -> str:
        """``dataset_id``, else the run's own id; it must have been prepared.

        Raises:
            ValueError: If neither id is given.
            FileNotFoundError: If prepare hasn't written that dataset.
        """
        dataset_id = dataset_id or run_id
        if not dataset_id:
            raise ValueError("Pass --dataset (or --run-id, if prepare used it)")
        if not (self.dataset_dir(dataset_id) / "manifest.json").exists():
            raise FileNotFoundError(
                f"No prepared dataset at {self.dataset_dir(dataset_id)}"
            )
        return dataset_id

    # Training
    def train_run(self, role: str, run_id: str) -> Path:
        """One training run of a role."""
        return self.base / "models" / role / run_id

    # Inference
    def infer_run(self, run_id: str) -> InferRun:
        """An ad-hoc inference run."""
        return InferRun(self.base / "infer" / run_id)

    def sweep_root(self, run_id: str) -> Path:
        """Parent of every val-sweep setting's inference run."""
        return self.infer_run(run_id).dir / "val_sweep"

    def sweep_infer_run(
        self, run_id: str, gap_fill_frames: int, det_conf: float
    ) -> InferRun:
        """Inference on val cameras at one (gap fill, det_conf) sweep setting."""
        return InferRun(
            self.sweep_root(run_id) / sweep_setting_name(gap_fill_frames, det_conf)
        )

    def test_infer_run(self, run_id: str) -> InferRun:
        """Inference on test cameras at the setting chosen on val."""
        return InferRun(self.infer_run(run_id).dir / "test")

    # Evaluation
    def evaluation_run(self, mode: str, run_id: str, role: str | None = None) -> Path:
        """Output of one evaluation mode, or of one role's model metrics."""
        return self.base / "eval" / run_id / (role if role is not None else mode)

    def slurm_logs(self, run_id: str) -> Path:
        """Job logs of one ``submit.sh`` run."""
        return self.base / "slurm_logs" / run_id
