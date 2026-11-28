"""Relabelers: map Classify export labels onto a label scheme before ground truth.

Picked by ``[prepare] relabel`` (fields from ``relabel_options``) and run by
prepare on the snapshot's Classify tasks, which stay as exported. A
relabeler's ``behaviors`` are the class vocabulary of everything downstream.
See ``docs/adr/0008-relabeling-and-split-schemes.md``. A new relabeler is a
:class:`Relabeler` subclass plus an entry in :data:`RELABELERS` and
:data:`RelabelName`.
"""

import copy
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from bird_cv.core.labels import Behaviors

Tasks = list[dict[str, Any]]


@dataclass(frozen=True)
class Relabeler:
    """Relabels each timeline range of a Classify export.

    Attributes:
        behaviors: The classes :meth:`relabel` outputs.
    """

    behaviors: ClassVar[Behaviors]

    def relabel(self, label: str, frame_range: dict[str, int]) -> str:
        """New label of one drawn range (inclusive bird-crop frames)."""
        raise NotImplementedError

    def __call__(self, tasks: Tasks) -> Tasks:
        """Copy of ``tasks`` with every timeline range relabeled.

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
                            by_label.setdefault(
                                self.relabel(label, frame_range), []
                            ).append(frame_range)
                    for i, (label, ranges) in enumerate(by_label.items()):
                        new = copy.deepcopy(result)
                        new["value"] |= {"timelinelabels": [label], "ranges": ranges}
                        if i:
                            new["id"] = f"{result.get('id')}_{i}"
                        results.append(new)
                annotation["result"] = results
        return tasks


@dataclass(frozen=True)
class Identity(Relabeler):
    """Keep the Classify project's labels."""

    behaviors: ClassVar[Behaviors] = Behaviors()

    def __call__(self, tasks: Tasks) -> Tasks:
        return tasks


@dataclass(frozen=True)
class ActivityGroups(Relabeler):
    """Group behaviors into ``restful``, ``awake`` and ``restless``.

    A ``head_move`` range is ``restful`` if at most
    ``head_move_restful_max_frames`` long, else ``awake``; the rest follow
    :attr:`groups`.
    """

    behaviors: ClassVar[Behaviors] = Behaviors(
        names=("awake", "restful", "restless"), resting="restful"
    )
    groups: ClassVar[dict[str, str]] = {
        "resting": "restful",
        "preening": "awake",
        "fluffing": "awake",
        "tail_wag": "awake",
        "flying": "restless",
        "walking___hopping": "restless",
        "flapping_wings": "restless",
    }
    head_move_restful_max_frames: int = 15

    def relabel(self, label: str, frame_range: dict[str, int]) -> str:
        if label == "head_move":
            length = frame_range["end"] - frame_range["start"] + 1
            return "restful" if length <= self.head_move_restful_max_frames else "awake"
        return self.groups[label]


RelabelName = Literal["identity", "activity_groups"]
"""Valid ``[prepare] relabel`` values; checked when the config loads."""

RELABELERS: dict[RelabelName, type[Relabeler]] = {
    "identity": Identity,
    "activity_groups": ActivityGroups,
}
"""Relabeler class by ``[prepare] relabel`` name."""
