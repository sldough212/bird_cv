"""Utilities for managing a Label Studio server and exporting annotations."""

from datetime import datetime
import logging
import socket
import subprocess
import time
from pathlib import Path


from label_studio_sdk import LabelStudio
from label_studio_sdk.errors import BadRequestError


logger = logging.getLogger(__name__)


def is_port_available(host: str, port: int) -> bool:
    """Check whether a TCP port is available for binding on a given host.

    This function attempts to bind to the specified host and port. If binding
    succeeds, the port is considered available.

    Args:
        host (str): Hostname or IP address to check (e.g., "localhost",
            "127.0.0.1").
        port (int): Port number to test.

    Returns:
        bool: True if the port is available, False if it is already in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_open_port(port: int, host: str) -> int:
    """Find the next available TCP port starting from a given port.

    Ports are checked sequentially, incrementing the port number until an
    available port is found.

    Args:
        port (int): Starting port number to check.
        host (str): Hostname or IP address on which to check port availability.

    Returns:
        int: The first available port number.
    """
    logger.debug("Searching for open port starting at %d on host %s", port, host)

    port_available = False
    while not port_available:
        port_available = is_port_available(port=port, host=host)
        if not port_available:
            logger.debug("Port %d unavailable, trying next", port)
            port += 1

    logger.info("Found open port: %d", port)
    return port


def close_server(port: int) -> None:
    """Terminate any process currently listening on the specified port.

    Processes bound to the given port are identified using `lsof` and
    forcefully terminated.

    Args:
        port (int): Port number whose listening processes should be stopped.
    """
    logger.info("Shutting down server on port %d", port)

    result = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
    )

    pids = result.stdout.strip().splitlines()
    for pid in pids:
        logger.debug("Killing process %s on port %d", pid, port)
        subprocess.run(["kill", "-9", pid])


def get_label_studio_client(
    host: str,
    port: int,
    api_key: str,
) -> LabelStudio:
    """Start a Label Studio server and return a connected client instance.

    This function launches Label Studio as a subprocess, waits for it to become
    available, and verifies connectivity using the provided API key.

    Args:
        host (str): Hostname or IP address to bind the Label Studio server.
        port (int): Port number on which to run Label Studio.
        api_key (str): API key used to authenticate the client.

    Returns:
        LabelStudio: An authenticated Label Studio client instance.

    Raises:
        RuntimeError: If the server does not become available within the
            connection timeout.
    """
    logger.info("Starting Label Studio at %s:%d", host, port)

    base_url = f"http://{host}:{port}"

    subprocess.Popen(
        [
            "label-studio",
            "start",
            "--port",
            str(port),
            "--host",
            host,
        ],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    for attempt in range(30):
        client = LabelStudio(base_url=base_url, api_key=api_key)
        try:
            client.users.whoami()
            logger.info("Label Studio connection verified")
            break
        except Exception:
            logger.debug("Label Studio not ready yet (attempt %d/30)", attempt + 1)
            time.sleep(1)
    else:
        logger.error("Failed to connect to Label Studio at %s", base_url)
        raise RuntimeError(f"Could not connect to label studio at {base_url}")

    return client


def export_label_studio_annotations(
    client: LabelStudio,
    project_id: int,
    output_path: Path,
    snapshot_title: str | None = None,
) -> None:
    """Export annotations for a Label Studio project to a JSON file.

    This function creates an export snapshot, verifies that the export has
    completed, and downloads the resulting JSON file to disk.

    Args:
        client (LabelStudio): Authenticated Label Studio client.
        project_id (int): ID of the project to export.
        output_path (Path): File path path where the exported JSON will be saved.
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
        serialization_options={"interpolate_key_frames": True},
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


def get_project_id_from_name(client: LabelStudio, project_name: str) -> int:
    """Retrieve a Label Studio project ID by its project name.

    Args:
        client (LabelStudio): Authenticated Label Studio client.
        project_name (str): Name (title) of the project.

    Returns:
        int: The ID of the matching project.

    Raises:
        ValueError: If no project with the given name is found.
    """
    logger.debug("Looking up project ID for project '%s'", project_name)

    projects = client.projects.list()
    for project in projects:
        if project.title == project_name:
            logger.info(
                "Found project '%s' with ID %d",
                project_name,
                project.id,
            )
            return project.id

    logger.error("Project '%s' not found", project_name)
    raise ValueError(f"Project '{project_name}' not found")


def get_label_studio_annotations(
    host: str,
    port: int,
    api_key: str,
    project_name: str,
    output_path: Path,
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
        snapshot_title=snapshot_title,
    )

    close_server(port=open_port)

    logger.info("Annotation export completed successfully")
