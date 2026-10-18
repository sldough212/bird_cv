"""Immutable, content-addressed snapshots of the Detect and Classify exports.

A snapshot is named by a hash of the annotation content only (each task's
video path and its result regions), so re-exporting unchanged labels gives
the same hash and no new snapshot. Ids, timestamps and other export
metadata are ignored.
"""

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bird_cv.client.get_label_studio_annotations import get_label_studio_annotations
from bird_cv.core.layout import snapshot_files
from bird_cv.core.manifest import new_manifest, write_manifest
from bird_cv.data.ground_truth import annotation_results

logger = logging.getLogger(__name__)


def _canonical_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce an export to sorted (video, regions) records for hashing.

    Only the annotation ground truth reads (the most recent non-cancelled
    one, see :func:`bird_cv.data.ground_truth.annotation_results`) counts.
    """
    # Region keys that carry annotation content; everything else (region ids,
    # origin, ...) is bookkeeping that changes without the labels changing.
    content_keys = ("type", "from_name", "to_name", "value")
    records = []
    for task in tasks:
        regions = [
            json.dumps({k: region.get(k) for k in content_keys}, sort_keys=True)
            for region in annotation_results(task)
        ]
        records.append({"video": task["data"]["video"], "regions": sorted(regions)})
    return sorted(records, key=lambda r: (r["video"], r["regions"]))


def annotation_hash(
    detect_tasks: list[dict[str, Any]],
    classify_tasks: list[dict[str, Any]],
    length: int = 16,
) -> str:
    """Hash the annotation content of a Detect and a Classify export.

    Args:
        detect_tasks: Parsed Detect project export (list of tasks).
        classify_tasks: Parsed Classify project export (list of tasks).
        length: Hex characters of the SHA-256 digest to keep.

    Returns:
        The first ``length`` hex characters of the SHA-256 digest.
    """
    content = json.dumps(
        {
            "detect": _canonical_tasks(detect_tasks),
            "classify": _canonical_tasks(classify_tasks),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:length]


def export_annotations(
    host: str,
    port: int,
    api_key: str,
    detect_project_name: str,
    classify_project_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Export the Detect (interpolated) and Classify projects from Label Studio.

    Args:
        host: Hostname or IP address for Label Studio.
        port: Starting port number to try.
        api_key: Label Studio API key.
        detect_project_name: Title of the Detect project.
        classify_project_name: Title of the Classify project.
        output_dir: Directory the two raw exports are written to.

    Returns:
        Paths to the Detect and Classify exports.
    """
    paths = snapshot_files(output_dir)
    for project_name, path in zip((detect_project_name, classify_project_name), paths):
        get_label_studio_annotations(
            host=host,
            port=port,
            api_key=api_key,
            project_name=project_name,
            output_path=path,
            interpolate_frames=True,
        )
    return paths


def create_snapshot(
    detect_export: Path,
    classify_export: Path,
    snapshots_dir: Path,
    *,
    snapshot_dir_for: Callable[[str], Path] | None = None,
) -> tuple[str, Path]:
    """Copy two exports into ``snapshots_dir/{hash}/`` unless that snapshot exists.

    Args:
        detect_export: Path to a Detect project JSON export.
        classify_export: Path to a Classify project JSON export.
        snapshots_dir: Root directory holding all snapshots.
        snapshot_dir_for: Maps a hash to its snapshot directory (e.g.
            :meth:`bird_cv.core.layout.Layout.snapshot`); overrides
            ``snapshots_dir/{hash}``.

    Returns:
        The snapshot hash and its directory, which contains ``detect.json``,
        ``classify.json`` and ``manifest.json``.
    """
    snapshot_hash = annotation_hash(
        json.loads(detect_export.read_text()), json.loads(classify_export.read_text())
    )
    snapshot_dir = (
        snapshot_dir_for(snapshot_hash)
        if snapshot_dir_for
        else snapshots_dir / snapshot_hash
    )
    if (snapshot_dir / "manifest.json").exists():
        logger.info("Snapshot %s already exists; labels unchanged", snapshot_hash)
        return snapshot_hash, snapshot_dir

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for export, copy in zip(
        (detect_export, classify_export), snapshot_files(snapshot_dir)
    ):
        shutil.copyfile(export, copy)
    write_manifest(
        new_manifest(
            "snapshot",
            snapshot_hash=snapshot_hash,
            inputs={"detect": str(detect_export), "classify": str(classify_export)},
        ),
        snapshot_dir,
    )
    logger.info("Wrote snapshot %s to %s", snapshot_hash, snapshot_dir)
    return snapshot_hash, snapshot_dir
