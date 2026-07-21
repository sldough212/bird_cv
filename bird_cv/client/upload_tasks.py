"""Upload local video files to a Label Studio project as tasks."""

import logging
from pathlib import Path

from bird_cv.client.utils import (
    find_open_port,
    get_label_studio_client,
    get_project_id_from_name,
    close_server,
)
from urllib.parse import quote

logger = logging.getLogger(__name__)


def upload_video_tasks(
    port: int,
    host: str,
    api_key: str,
    project_name: str,
    video_path: Path,
):
    """Upload every video under a directory as a new Label Studio task.

    Recursively scans `video_path` for video files, builds a Label Studio
    local-files task for each one, skips any video already imported into
    the project, and imports the rest.

    Args:
        port (int): Preferred port to start Label Studio on; if unavailable,
            the next open port is used instead.
        host (str): Hostname or IP address for Label Studio.
        api_key (str): API key used to authenticate with Label Studio.
        project_name (str): Name (title) of the Label Studio project to
            upload tasks to.
        video_path (Path): Root directory to recursively search for video
            files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).

    Returns:
        None
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

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    tasks = [
        {"video": f"/data/local-files/?d={quote(str(path.relative_to('/')))}"}
        for path in video_path.rglob("*")
        if path.suffix.lower() in video_extensions
    ]

    # Check task not already in project
    existing_tasks = list(client.tasks.list(project=project_id))
    existing_videos = {t.data["video"] for t in existing_tasks}
    new_tasks = [task for task in tasks if task["video"] not in existing_videos]
    logger.info(f"Found {len(new_tasks)} new tasks to import")

    client.projects.import_tasks(id=project_id, request=new_tasks)

    close_server(port=open_port)
