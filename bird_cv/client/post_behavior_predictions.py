"""Generate VideoMAE behavior predictions and post them to Label Studio."""

import logging
from pathlib import Path

from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

from bird_cv.classification.predict_video_model import (
    load_videomae_model,
    predict_video_segments,
)
from bird_cv.client.utils import (
    close_server,
    find_open_port,
    get_label_studio_client,
    get_local_path,
    get_project_id_from_name,
)

logger = logging.getLogger(__name__)


def get_videomae_predictions(
    model_path: Path | str,
    tasks_to_label: list,
) -> ModelResponse:
    """Run VideoMAE behavior classification over Label Studio tasks and build predictions.

    Loads a VideoMAE model, runs windowed inference over each task's video, and
    converts the resulting behavior segments into Label Studio `timelinelabels`
    regions suitable for submission as predictions.

    Args:
        model_path (Path | str): Path to the VideoMAE model weights.
        tasks_to_label (list): Label Studio tasks (as dicts, e.g. from
            `Task.model_dump()`) whose `data["video"]` field points to a
            video to run predictions on.

    Returns:
        ModelResponse: Predictions for each task, in the same order as
            `tasks_to_label`.
    """
    model, processor, dev, id_to_label = load_videomae_model(str(model_path))

    predictions = []
    for t in tasks_to_label:
        local_path = get_local_path(t["data"]["video"])

        segments = predict_video_segments(
            local_path, model, processor, dev, id_to_label
        )

        result = [
            {
                "from_name": "behavior",
                "to_name": "video",
                "type": "timelinelabels",
                "value": {
                    "timelinelabels": [label],
                    "ranges": [
                        {"start": start_frame, "end": end_frame, "timetype": "frames"}
                    ],
                },
            }
            for label, start_frame, end_frame in segments
        ]

        predictions.append(PredictionValue(result=result))

    response = ModelResponse(predictions=predictions, model_version=str(model_path))

    return response


def post_behavior_predictions(
    host: str,
    port: int,
    api_key: str,
    project_name: str,
    model_path: Path | str,
    model_version: str,
    predict_labeled: bool = False,
) -> None:
    """Generate VideoMAE behavior predictions for a project and post them to Label Studio.

    Starts (or connects to) a Label Studio server, fetches unlabeled tasks
    for the given project, runs the VideoMAE behavior classification model
    on each task, and submits the resulting predictions back to Label Studio.

    Args:
        host (str): Hostname or IP address for Label Studio.
        port (int): Preferred port to start Label Studio on; if unavailable,
            the next open port is used instead.
        api_key (str): API key for Label Studio authentication.
        project_name (str): Name (title) of the project to generate
            predictions for.
        model_path (Path | str): Path to the VideoMAE model weights used for
            prediction.
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

    response = get_videomae_predictions(
        model_path=model_path,
        tasks_to_label=tasks_to_label,
    )

    for task, pred in zip(tasks_to_label, response.predictions):
        result = pred["result"] if isinstance(pred, dict) else pred.result
        client.predictions.create(
            task=task["id"], result=result, model_version=model_version
        )

        logger.info(f"Task {task['id']} prediction posted")

    # Close the port
    close_server(port=open_port)
