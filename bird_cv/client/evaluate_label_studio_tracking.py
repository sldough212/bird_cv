"""Compute tracking metrics for Label Studio tasks with both a submitted
annotation and a posted model prediction."""

import logging
from pathlib import Path

import polars as pl

from bird_cv.client.utils import (
    close_server,
    find_open_port,
    get_label_studio_client,
    get_project_id_from_name,
)
from bird_cv.detection.mot_evaluate import compute_mot_metrics, ls_regions_to_mot

logger = logging.getLogger(__name__)


def _select_annotation(
    annotations: list[dict], annotation_id: int | None = None
) -> dict | None:
    """Pick which submitted annotation on a task to use as ground truth.

    Cancelled (skipped/rejected) annotations are never eligible.

    Args:
        annotations: A task's ``annotations`` list.
        annotation_id: If given, return that specific annotation (or None
            if it isn't on the task, or was cancelled). Otherwise the most
            recently created annotation is used.

    Returns:
        The selected annotation dict, or None if none is eligible.
    """
    candidates = [a for a in annotations if not a.get("was_cancelled")]
    if annotation_id is not None:
        return next((a for a in candidates if a["id"] == annotation_id), None)
    if not candidates:
        return None
    return max(candidates, key=lambda a: a["created_at"])


def _select_prediction(
    predictions: list[dict],
    prediction_id: int | None = None,
    model_version: str | None = None,
) -> dict | None:
    """Pick which posted prediction on a task to score against ground truth.

    Args:
        predictions: A task's ``predictions`` list.
        prediction_id: If given, return that specific prediction (or None
            if it isn't on the task). Takes precedence over
            `model_version`.
        model_version: If given (and `prediction_id` is not), restrict to
            predictions with this model version before picking the most
            recent one.

    Returns:
        The selected prediction dict, or None if none is eligible.
    """
    if prediction_id is not None:
        return next((p for p in predictions if p["id"] == prediction_id), None)

    candidates = predictions
    if model_version is not None:
        candidates = [p for p in candidates if p.get("model_version") == model_version]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p["created_at"])


def evaluate_label_studio_tracking(
    host: str,
    port: int,
    api_key: str,
    project_name: str,
    output_path: Path,
    annotation_id: int | None = None,
    prediction_id: int | None = None,
    model_version: str | None = None,
) -> None:
    """Compute MOT metrics for Label Studio tasks with both a submitted
    annotation and a posted model prediction.

    For each task in the project that has at least one non-cancelled
    annotation and at least one prediction, selects a ground-truth
    annotation and a prediction to score against it (by default the most
    recent of each — see `annotation_id`/`prediction_id`/`model_version`
    to pin specific ones instead), converts both to MOT format, and
    computes tracking metrics. Tasks with no eligible annotation/
    prediction pair are skipped.

    Args:
        host: Hostname or IP address of the Label Studio server.
        port: Preferred port to start/connect to Label Studio on; if
            unavailable, the next open port is used instead.
        api_key: API key used to authenticate with Label Studio.
        project_name: Name (title) of the Label Studio project to
            evaluate.
        output_path: Path where results will be saved as a parquet file.
        annotation_id: If given, use this specific annotation as ground
            truth on every task that has it, instead of each task's most
            recent annotation.
        prediction_id: If given, use this specific prediction as the one
            to score on every task that has it, instead of each task's
            most recent prediction. Takes precedence over `model_version`.
        model_version: If given (and `prediction_id` is not), only
            consider predictions with this model version, using the most
            recent match per task.
    """
    logger.info("Starting tracking evaluation for project '%s'", project_name)

    open_port = find_open_port(port=port, host=host)
    client = get_label_studio_client(host=host, port=open_port, api_key=api_key)
    project_id = get_project_id_from_name(client=client, project_name=project_name)

    tasks = client.tasks.list(project=project_id, fields="all")

    records = []
    for task in tasks:
        task = task.model_dump()
        if not task.get("annotations") or not task.get("predictions"):
            continue

        annotation = _select_annotation(task["annotations"], annotation_id)
        prediction = _select_prediction(
            task["predictions"], prediction_id, model_version
        )
        if annotation is None or prediction is None:
            logger.debug(
                "Task %s: no matching annotation/prediction, skipping", task["id"]
            )
            continue

        gt = ls_regions_to_mot(annotation["result"])
        pred = ls_regions_to_mot(prediction["result"])
        metrics = compute_mot_metrics(gt=gt, pred=pred)

        records.append(
            {
                "task_id": task["id"],
                "annotation_id": annotation["id"],
                "prediction_id": prediction.get("id"),
                "model_version": prediction.get("model_version"),
            }
            | metrics
        )

    close_server(port=open_port)

    results = pl.DataFrame(records)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    results.write_parquet(output_path)

    logger.info("Tracking evaluation saved to %s", output_path)
