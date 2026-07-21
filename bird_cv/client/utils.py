"""Utilities for managing a Label Studio server and exporting annotations."""

import logging
import socket
import subprocess
import time
from pathlib import Path

from urllib.parse import unquote

import cv2

from label_studio_sdk import LabelStudio


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


def get_local_path(video_url: str) -> Path:
    """Resolve a Label Studio local-files video URL to an absolute filesystem path.

    Args:
        video_url (str): Label Studio video URL in the
            `/data/local-files/?d=<url-encoded-path>` format.

    Returns:
        Path: Absolute local filesystem path to the video file.
    """
    local_url = video_url.split("?d=")[-1]
    return Path("/" + unquote(local_url))


def get_video_info(path: Path | str) -> tuple[int, float]:
    """Get the frame count and duration of a video file.

    Args:
        path (Path | str): Path to the video file.

    Returns:
        tuple[int, float]: The total frame count and duration in seconds.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames_count, frames_count / fps


def process_lifespans(sequence: list[dict]) -> list[dict]:
    """Disable trailing keyframes at gaps in a videorectangle track.

    Given a `sequence` of per-frame region dicts sorted by increasing
    `frame`, marks a keyframe as `enabled: False` whenever the next keyframe
    isn't on the immediately following frame (i.e. tracking was lost), and
    always disables the final keyframe. This keeps Label Studio from
    interpolating a track across a gap or past its last confirmed frame.

    Args:
        sequence (list[dict]): Per-frame region dicts, each containing at
            least `frame` and `enabled` keys, sorted by increasing frame
            number.

    Returns:
        list[dict]: The same sequence, mutated in place, with `enabled`
            flags updated.
    """
    for i in range(1, len(sequence)):
        if sequence[i]["frame"] - sequence[i - 1]["frame"] > 1:
            sequence[i - 1]["enabled"] = False
    if sequence:
        sequence[-1]["enabled"] = False
    return sequence
