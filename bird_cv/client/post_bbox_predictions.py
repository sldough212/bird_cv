"""Generate YOLO bounding-box predictions and post them to Label Studio."""

import logging
from collections import defaultdict
from pathlib import Path

from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from ultralytics import YOLO

from bird_cv.client.utils import (
    close_server,
    find_open_port,
    get_label_studio_client,
    get_local_path,
    get_project_id_from_name,
    get_video_info,
    process_lifespans,
)

logger = logging.getLogger(__name__)


def get_yolo_predictions(
    model_path: Path | str,
    tracker_path: Path | str,
    tasks_to_label: list,
) -> ModelResponse:
    """Run YOLO tracking over a set of Label Studio tasks and build predictions.

    Loads a YOLO model, tracks objects across each task's video, and converts
    the resulting per-frame tracks into Label Studio `videorectangle` regions
    suitable for submission as predictions.

    Args:
        model_path (Path | str): Path to the YOLO model weights (e.g. a
            `.pt` file).
        tracker_path (Path | str): Path to the tracker config (e.g.
            `tracker.yaml`) passed to `model.track`.
        tasks_to_label (list): Label Studio tasks (as dicts, e.g. from
            `Task.model_dump()`) whose `data["video"]` field points to a
            video to run predictions on.

    Returns:
        ModelResponse: Predictions for each task, in the same order as
            `tasks_to_label`.
    """

    # Set the yolo model
    model_path = str(model_path)
    model = YOLO(model_path)
    model_names = model.names

    predictions = []
    for t in tasks_to_label:
        local_path = get_local_path(t["data"]["video"])

        frames_count, duration = get_video_info(local_path)
        results = model.track(
            local_path,
            stream=True,
            conf=0.05,
            iou=0.70,
            tracker=tracker_path,
        )

        tracks = defaultdict(list)
        track_labels = {}
        for frame_idx, result in enumerate(results):
            data = result.boxes
            if not data.is_track:
                continue
            for i, track_id in enumerate(data.id.tolist()):
                score = float(data.conf[i])
                x, y, w, h = data.xywhn[i].tolist()
                track_labels[track_id] = model_names[int(data.cls[i])]
                tracks[track_id].append(
                    {
                        "frame": frame_idx + 1,
                        "enabled": True,
                        "rotation": 0,
                        "x": (x - w / 2) * 100,
                        "y": (y - h / 2) * 100,
                        "width": w * 100,
                        "height": h * 100,
                        "time": (frame_idx + 1) * (duration / frames_count),
                        "score": score,
                    }
                )

        regions = []
        for track_id, sequence in tracks.items():
            sequence = process_lifespans(sequence)
            regions.append(
                {
                    "from_name": "box",
                    "to_name": "video",
                    "type": "videorectangle",
                    "value": {
                        "framesCount": frames_count,
                        "duration": duration,
                        "sequence": sequence,
                        "labels": [track_labels[track_id]],
                    },
                    "score": max(f["score"] for f in sequence),
                    "origin": "manual",
                }
            )

        predictions.append(PredictionValue(result=regions))

    response = ModelResponse(predictions=predictions, model_version=model_path)

    return response


def post_bbox_predictions(
    host: str,
    port: int,
    api_key: str,
    project_name: str,
    model_path: Path,
    tracker_path: Path,
    model_version: str,
    predict_labeled: bool = False,
) -> None:
    """Generate YOLO bbox predictions for a project and post them to Label Studio.

    Starts (or connects to) a Label Studio server, fetches tasks for the
    given project, runs YOLO tracking on each task's video, and submits the
    resulting regions back as predictions.

    Args:
        host (str): Hostname or IP address of the Label Studio server.
        port (int): Preferred port to start Label Studio on; if unavailable,
            the next open port is used instead.
        api_key (str): API key used to authenticate with Label Studio.
        project_name (str): Name (title) of the Label Studio project to
            generate predictions for.
        model_path (Path): Path to the YOLO model weights used for
            prediction.
        tracker_path (Path): Path to the tracker config used by `model.track`.
        model_version (str): Version string recorded on each created
            prediction.
        predict_labeled (bool): If True, generate predictions for all tasks,
            including ones that already have annotations. If False (default),
            only unlabeled tasks are predicted on.

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

    # Get unlabeled tasks
    tasks = client.tasks.list(project=project_id)

    if predict_labeled:
        tasks_to_label = [t.model_dump() for t in tasks]
    else:
        tasks_to_label = [t.model_dump() for t in tasks if not t.annotations]

    logger.info(f"Predicting on {len(tasks_to_label)} tasks")

    response = get_yolo_predictions(
        model_path=model_path,
        tracker_path=tracker_path,
        tasks_to_label=tasks_to_label,
    )

    for task, pred in zip(tasks_to_label, response.predictions):
        result = pred["result"] if isinstance(pred, dict) else pred.result
        client.predictions.create(
            task=task["id"],
            result=result,
            model_version=model_version,
        )

        logger.info(f"Task {task['id']} prediction posted")

    # Close port
    close_server(port=open_port)
