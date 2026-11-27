"""The class vocabulary shared by ground truth and predictions.

See ``docs/adr/0003-no-bird-class.md``. Built from the ``[behaviors]`` config
table (``BirdCVConfig.behaviors``); behavior names match the Classify
project's ``timelinelabels`` exactly, so no translation is needed when
parsing exports.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Behaviors:
    """The annotated behaviors plus the pipeline's ``no_bird`` class.

    Attributes:
        names: Behaviors annotated in the Classify project, sorted.
        resting: The behavior sampling caps (one of ``names``).
        no_bird: Class for a canonical frame with no bird to classify.
    """

    names: tuple[str, ...] = field(
        default=(
            "flapping_wings",
            "fluffing",
            "flying",
            "head_move",
            "preening",
            "resting",
            "tail_wag",
            "walking___hopping",
        )
    )
    resting: str = "resting"
    no_bird: str = "no_bird"

    def __post_init__(self) -> None:
        names = tuple(self.names)
        object.__setattr__(self, "names", names)
        if list(names) != sorted(set(names)):
            raise ValueError(f"Behavior names must be unique and sorted: {names}")
        if self.resting not in names:
            raise ValueError(f"resting {self.resting!r} is not one of {names}")
        if self.no_bird in names:
            raise ValueError(f"no_bird {self.no_bird!r} can't be a behavior")

    @property
    def classes(self) -> tuple[str, ...]:
        """Every class, in the order used for confusion matrices and reports."""
        return (*self.names, self.no_bird)

    @property
    def active(self) -> tuple[str, ...]:
        """Every behavior except resting; used for sampling caps."""
        return tuple(b for b in self.names if b != self.resting)

    def validate(self, label: str) -> str:
        """Return ``label`` unchanged if it's a known class.

        Raises:
            ValueError: If ``label`` is not in :attr:`classes`.
        """
        if label not in self.classes:
            raise ValueError(f"Unknown class {label!r}; expected one of {self.classes}")
        return label
