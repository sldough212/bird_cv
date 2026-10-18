"""Every output path under ``path_to_base_output``, defined once.

All new pipeline stages (prepare, dataset build, train, infer, evaluate) and
the slurm scripts locate their inputs and outputs through :class:`Layout`, so
a path changes here or nowhere::

    annotations/{stamp}/                      Label Studio exports
    snapshots/{snapshot_hash}/                detect.json, classify.json, manifest.json
    splits/split_v{n}.parquet                 SPLIT_SCHEMA
    splits/split_v{n}/manifest.json
    splits/cv_folds_v{n}.parquet              CV_FOLD_SCHEMA
    prepared/current.json                     the prepared data later stages use
    prepared/{snapshot_hash}/gt_v{g}/         ground_truth.parquet, ground_truth_stats.json
        split_v{n}/sampling_{cfghash}/        split_summary, yolo_frames, videomae_clips
                                              (.parquet), manifest.json
            datasets/{role}/{kind}/frozen/    training data on the frozen split
            datasets/{role}/{kind}/fold{k}/   one CV fold, with its split.parquet
    train/{role}/{run_id}/                    frozen-split training run
    train/{role}/fold{k}/{run_id}/            CV fold training run
    infer/{run_id}/                           an ad-hoc inference run, or the root of:
        val_sweep/gap{g}_conf{c}/             one val-sweep setting
        test/                                 test cameras at the chosen setting
        cv/fold{k}/                           one CV fold's test cameras
    evaluation/{mode}/{run_id}/[{role}/]

Every inference run directory has the layout of :class:`InferRun`. The
legacy Label Studio pipelines keep their own paths
(``BirdCVConfig.__post_init__``).
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


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


def cv_folds_file(splits_dir: Path, version: int) -> Path:
    """CV folds belonging to split version ``version``."""
    return splits_dir / f"cv_folds_v{version}.parquet"


def sweep_setting_name(gap_fill_frames: int, det_conf: float) -> str:
    """Directory name of one val-sweep setting, e.g. ``gap8_conf0.25``."""
    return f"gap{gap_fill_frames}_conf{det_conf:g}"


@dataclass(frozen=True)
class Prepared:
    """Identifies one prepare output: the data every later stage reads.

    Attributes:
        snapshot_hash: Label Studio snapshot.
        gt_version: Ground-truth rules version
            (:data:`bird_cv.data.ground_truth.GT_VERSION`).
        split_version: Frozen split version.
        sampling_hash: Hash of the config values frame and clip selection read.
    """

    snapshot_hash: str
    gt_version: int
    split_version: int
    sampling_hash: str


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

    # Label Studio exports and snapshots

    def annotation_exports(self, stamp: str) -> Path:
        """Directory one Label Studio export run writes into."""
        return self.base / "annotations" / stamp

    @property
    def snapshots(self) -> Path:
        """Root of every snapshot, ``{snapshot_hash}/`` below it."""
        return self.base / "snapshots"

    def snapshot(self, snapshot_hash: str) -> Path:
        """One snapshot: ``detect.json``, ``classify.json``, ``manifest.json``."""
        return self.snapshots / snapshot_hash

    # Splits

    @property
    def splits(self) -> Path:
        """Directory of every split version and its CV folds."""
        return self.base / "splits"

    def split(self, version: int) -> Path:
        """Split version ``version`` (``SPLIT_SCHEMA``)."""
        return split_file(self.splits, version)

    def cv_folds(self, version: int) -> Path:
        """CV folds of split version ``version`` (``CV_FOLD_SCHEMA``)."""
        return cv_folds_file(self.splits, version)

    # Prepared data

    def ground_truth_dir(self, snapshot_hash: str, gt_version: int) -> Path:
        """Ground truth built from one snapshot under one rules version."""
        return self.base / "prepared" / snapshot_hash / f"gt_v{gt_version}"

    def ground_truth(self, snapshot_hash: str, gt_version: int) -> Path:
        """Ground truth table (``GROUND_TRUTH_SCHEMA``)."""
        return self.ground_truth_dir(snapshot_hash, gt_version) / "ground_truth.parquet"

    def ground_truth_stats(self, snapshot_hash: str, gt_version: int) -> Path:
        """Counts of what the ground-truth conversion fixed or dropped."""
        return (
            self.ground_truth_dir(snapshot_hash, gt_version) / "ground_truth_stats.json"
        )

    def sampling_dir(self, prepared: Prepared) -> Path:
        """Summary and sampled frame/clip lists of one prepare output."""
        return (
            self.ground_truth_dir(prepared.snapshot_hash, prepared.gt_version)
            / f"split_v{prepared.split_version}"
            / f"sampling_{prepared.sampling_hash}"
        )

    def split_summary(self, prepared: Prepared) -> Path:
        """Per-split class counts (``SPLIT_SUMMARY_SCHEMA``)."""
        return self.sampling_dir(prepared) / "split_summary.parquet"

    def yolo_frames(self, prepared: Prepared) -> Path:
        """Sampled detector training frames (``YOLO_FRAMES_SCHEMA``)."""
        return self.sampling_dir(prepared) / "yolo_frames.parquet"

    def videomae_clips(self, prepared: Prepared) -> Path:
        """Sampled classifier training clips (``VIDEOMAE_CLIPS_SCHEMA``)."""
        return self.sampling_dir(prepared) / "videomae_clips.parquet"

    def dataset(
        self, prepared: Prepared, role: str, kind: str, fold: int | None = None
    ) -> Path:
        """Training data of one role's model kind: frozen split or CV fold ``fold``."""
        root = self.sampling_dir(prepared) / "datasets" / role / kind
        return root / ("frozen" if fold is None else f"fold{fold}")

    def dataset_split(
        self, prepared: Prepared, role: str, kind: str, fold: int | None = None
    ) -> Path:
        """The camera split a dataset was built from (``camera_id``, ``split``)."""
        return self.dataset(prepared, role, kind, fold) / "split.parquet"

    @property
    def current_prepared(self) -> Path:
        """Pointer to the prepare output later stages read."""
        return self.base / "prepared" / "current.json"

    def write_current(self, prepared: Prepared) -> Path:
        """Point later stages at ``prepared``; returns the pointer file."""
        self.current_prepared.parent.mkdir(parents=True, exist_ok=True)
        self.current_prepared.write_text(json.dumps(asdict(prepared), indent=2))
        return self.current_prepared

    def read_current(self) -> Prepared:
        """The prepare output later stages read.

        Raises:
            FileNotFoundError: If prepare has not run under this base.
        """
        if not self.current_prepared.exists():
            raise FileNotFoundError(
                f"No {self.current_prepared}; run bird_cv/pipelines/prepare.py first"
            )
        return Prepared(**json.loads(self.current_prepared.read_text()))

    # Training

    def train_run(self, role: str, run_id: str, fold: int | None = None) -> Path:
        """One training run of a role, on the frozen split or a CV fold."""
        root = self.base / "train" / role
        return (root if fold is None else root / f"fold{fold}") / run_id

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

    def cv_root(self, run_id: str) -> Path:
        """Parent of every CV fold's inference run."""
        return self.infer_run(run_id).dir / "cv"

    def cv_infer_run(self, run_id: str, fold: int) -> InferRun:
        """Inference on one CV fold's test cameras with that fold's models."""
        return InferRun(self.cv_root(run_id) / f"fold{fold}")

    # Evaluation

    def evaluation_run(self, mode: str, run_id: str, role: str | None = None) -> Path:
        """Output of one evaluation mode (and role, for model metrics)."""
        out = self.base / "evaluation" / mode / run_id
        return out / role if role is not None else out
