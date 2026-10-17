"""The class vocabulary shared by ground truth and predictions.

See ``docs/adr/0003-no-bird-class.md``. Behavior names match the Classify
project's ``timelinelabels`` exactly, so no translation is needed when
parsing exports.
"""

BEHAVIORS: tuple[str, ...] = (
    "flapping_wings",
    "fluffing",
    "flying",
    "head_move",
    "preening",
    "resting",
    "tail_wag",
    "walking___hopping",
)
"""The eight behaviors annotated in the Classify project, sorted."""

RESTING = "resting"

NO_BIRD = "no_bird"
"""Class for a canonical frame with no bird to classify."""

CLASSES: tuple[str, ...] = BEHAVIORS + (NO_BIRD,)
"""All nine classes, in the order used for confusion matrices and reports."""

ACTIVE_BEHAVIORS: tuple[str, ...] = tuple(b for b in BEHAVIORS if b != RESTING)
"""Every behavior except resting; used for sampling caps and the later
combined-"active" experiment."""


def validate_class(label: str) -> str:
    """Return ``label`` unchanged if it's a known class, else raise.

    Args:
        label: A class name.

    Returns:
        The same label.

    Raises:
        ValueError: If ``label`` is not in :data:`CLASSES`.
    """
    if label not in CLASSES:
        raise ValueError(f"Unknown class {label!r}; expected one of {CLASSES}")
    return label
