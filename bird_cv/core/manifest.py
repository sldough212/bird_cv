"""Run manifests: what produced a dataset, model or set of predictions.

Every prepared dataset, training run, inference run and evaluation writes a
``manifest.json`` so any output traces back to the labels, split and code
that produced it.
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import msgspec


class Manifest(msgspec.Struct):
    """Provenance for one pipeline output directory.

    Attributes:
        kind: What produced the output, e.g. ``"prepare"``, ``"train_yolo"``,
            ``"train_videomae"``, ``"infer"``, ``"evaluate"``.
        git_commit: Commit of the code that ran, suffixed ``-dirty`` if the
            working tree had uncommitted changes.
        created_at: ISO-8601 local timestamp.
        snapshot_hash: Hash of the Label Studio snapshot used, if any.
        split_version: Split version used, if any.
        config: The resolved configuration for the run.
        inputs: Paths to upstream artifacts (models, datasets) the run read.
    """

    kind: str
    git_commit: str
    created_at: str
    snapshot_hash: str | None = None
    split_version: int | None = None
    config: dict[str, Any] = {}
    inputs: dict[str, str] = {}


def current_git_commit(repo_dir: Path | None = None) -> str:
    """Return ``HEAD``'s commit hash, suffixed ``-dirty`` for uncommitted changes.

    Defaults to the checkout this package lives in, so the result doesn't
    depend on the working directory a job was submitted from. Returns
    ``"unknown"`` outside a git checkout.
    """
    cwd = str(repo_dir or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def new_manifest(
    kind: str,
    snapshot_hash: str | None = None,
    split_version: int | None = None,
    config: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
) -> Manifest:
    """Build a manifest stamped with the current commit and time."""
    return Manifest(
        kind=kind,
        git_commit=current_git_commit(),
        created_at=datetime.now().isoformat(timespec="seconds"),
        snapshot_hash=snapshot_hash,
        split_version=split_version,
        config=config or {},
        inputs=inputs or {},
    )


def write_manifest(manifest: Manifest, output_dir: Path) -> Path:
    """Write ``manifest.json`` into ``output_dir`` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"

    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Unsupported type: {type(obj)}")

    path.write_bytes(
        msgspec.json.format(msgspec.json.encode(manifest, enc_hook=enc_hook))
    )
    return path


def read_manifest(output_dir: Path) -> Manifest:
    """Read the ``manifest.json`` in ``output_dir``."""
    return msgspec.json.decode(
        (output_dir / "manifest.json").read_bytes(), type=Manifest
    )
