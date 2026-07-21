"""Utilities for managing a Label Studio server and exporting annotations."""

import logging
from datetime import datetime
from pathlib import Path

from label_studio_sdk import LabelStudio
from label_studio_sdk.errors import BadRequestError

from bird_cv.client.utils import (
    close_server,
    find_open_port,
    get_label_studio_client,
    get_project_id_from_name,
)


logger = logging.getLogger(__name__)


def export_label_studio_annotations(
    client: LabelStudio,
    project_id: int,
    output_path: Path,
    interpolate_frames: bool = True,
    snapshot_title: str | None = None,
) -> None:
    """Export annotations for a Label Studio project to a JSON file.

    This function creates an export snapshot, verifies that the export has
    completed, and downloads the resulting JSON file to disk.

    Args:
        client (LabelStudio): Authenticated Label Studio client.
        project_id (int): ID of the project to export.
        output_path (Path): File path path where the exported JSON will be saved.
        interpolate_frames (bool): Whether to interpolate frames in the exported JSON.
        snapshot_title (st | None): Name assigned to exported snapshot.

    Raises:
        BadRequestError: If the export snapshot is not completed or is not ready.
    """
    logger.info("Creating export snapshot for project %d", project_id)

    if not snapshot_title:
        # Get current date and time
        now = datetime.now()

        # Format suitable for filename: "YYYY-MM-DD_HH-MM-SS"
        snapshot_title = now.strftime("%Y-%m-%d_%H-%M-%S")

    export = client.projects.exports.create(
        id=project_id,
        title=snapshot_title,
        serialization_options={"interpolate_key_frames": interpolate_frames},
        task_filter_options={"only_with_annotations": True},
    )
    export_id = export.id

    job = client.projects.exports.get(id=project_id, export_pk=export_id)
    if job.status != "completed":
        logger.error("Export snapshot not ready (status=%s)", job.status)
        raise BadRequestError(
            status_code=409,
            body=f"Export not ready: {job.status}",
        )

    logger.info("Downloading annotations to %s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in client.projects.exports.download(
            id=project_id,
            export_pk=export_id,
            export_type="JSON",
            request_options={"chunk_size": 1024},
        ):
            f.write(chunk)

    logger.info("Annotations saved to %s", output_path)


def get_label_studio_annotations(
    host: str,
    port: int,
    api_key: str,
    project_name: str,
    output_path: Path,
    interpolate_frames: bool = True,
    snapshot_title: str | None = None,
) -> None:
    """Export Label Studio annotations for a project by name.

    This function finds an open port, launches a Label Studio server,
    resolves a project ID from its name, exports annotations to disk,
    and shuts down the server.

    Args:
        host (str): Hostname or IP address for Label Studio.
        port (int): Starting port number to try.
        api_key (str): API key for Label Studio authentication.
        project_name (str): Name (title) of the project to export.
        output_path (Path): Destination file path for exported annotations.
        interpolate_frames (bool): Whether to interpolate frames in the exported JSON.
        snapshot_title (str | None): Name assigned to exported snapshot.
    """
    logger.info("Starting annotation export for project '%s'", project_name)

    open_port = find_open_port(port=port, host=host)

    client = get_label_studio_client(
        host=host,
        port=open_port,
        api_key=api_key,
    )

    project_id = get_project_id_from_name(
        client=client,
        project_name=project_name,
    )

    export_label_studio_annotations(
        client=client,
        project_id=project_id,
        output_path=output_path,
        interpolate_frames=interpolate_frames,
        snapshot_title=snapshot_title,
    )

    close_server(port=open_port)

    logger.info("Annotation export completed successfully")
