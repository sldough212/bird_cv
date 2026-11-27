"""Relabelers: map Classify export labels onto a label scheme before ground truth.

Picked by ``[prepare] relabel`` (keyword arguments from ``relabel_options``)
and run by prepare on the snapshot's Classify tasks, which stay as exported.
See ``docs/adr/0008-relabeling-and-split-schemes.md``. The relabeled names
must be ``[behaviors] names``.
"""

import copy
from collections.abc import Callable
from typing import Any

Tasks = list[dict[str, Any]]


def identity(tasks: Tasks) -> Tasks:
    """Keep the Classify project's labels."""
    return tasks


ACTIVITY_GROUPS = {
    "resting": "restful",
    "preening": "awake",
    "fluffing": "awake",
    "tail_wag": "awake",
    "flying": "restless",
    "walking___hopping": "restless",
    "flapping_wings": "restless",
}
"""Group of every behavior but ``head_move``, which goes by length."""


def activity_groups(tasks: Tasks, head_move_restful_max_frames: int = 15) -> Tasks:
    """Group behaviors into ``restful``, ``awake`` and ``restless``.

    A ``head_move`` range (one drawn span, inclusive bird-crop frames) is
    ``restful`` if at most ``head_move_restful_max_frames`` long, else
    ``awake``; the rest follow :data:`ACTIVITY_GROUPS`.
    """

    def group(label: str, frame_range: dict[str, int]) -> str:
        if label == "head_move":
            length = frame_range["end"] - frame_range["start"] + 1
            return "restful" if length <= head_move_restful_max_frames else "awake"
        return ACTIVITY_GROUPS[label]

    return relabel_ranges(tasks, group)


def relabel_ranges(
    tasks: Tasks, relabel: Callable[[str, dict[str, int]], str]
) -> Tasks:
    """Copy of ``tasks`` with every timeline range relabeled by ``relabel``.

    A ``timelinelabels`` result becomes one result per new label, holding
    the ranges given that label; other results are kept as they are.
    """
    tasks = copy.deepcopy(tasks)
    for task in tasks:
        for annotation in task.get("annotations", []):
            results = []
            for result in annotation.get("result", []):
                if result.get("type") != "timelinelabels":
                    results.append(result)
                    continue
                by_label: dict[str, list[dict[str, int]]] = {}
                for label in result["value"]["timelinelabels"]:
                    for frame_range in result["value"]["ranges"]:
                        by_label.setdefault(relabel(label, frame_range), []).append(
                            frame_range
                        )
                for i, (label, ranges) in enumerate(by_label.items()):
                    new = copy.deepcopy(result)
                    new["value"] |= {"timelinelabels": [label], "ranges": ranges}
                    if i:
                        new["id"] = f"{result.get('id')}_{i}"
                    results.append(new)
            annotation["result"] = results
    return tasks


RELABELERS: dict[str, Callable[..., Tasks]] = {
    "identity": identity,
    "activity_groups": activity_groups,
}
"""Relabelers by ``[prepare] relabel`` name."""
