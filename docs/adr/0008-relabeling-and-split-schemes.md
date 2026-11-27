# Relabel before ground truth; one split lineage per label scheme

Eight behaviors are too many and too rare for VideoMAE to learn from our label counts, and labelers disagree on the fine distinctions. Prepare therefore runs a **relabeler** (`[prepare] relabel`, `bird_cv/preprocessing/relabel.py`) on the Classify export before ground truth, and everything downstream (split, sampling, training, evaluation) sees only the relabeled classes. `identity` keeps the Classify labels; `activity_groups` maps them to `restful` (resting, and each drawn `head_move` span of at most `head_move_restful_max_frames` = 15 bird-crop frames), `awake` (longer head moves, preening, fluffing, tail wag) and `restless` (flying, walking/hopping, flapping wings). The snapshot stays as exported, labelers keep labeling the fine classes, and the Label Studio pipelines are unchanged.

A **label scheme** (relabeler plus `[behaviors] names`) changes which cameras hold which classes, so it gets its own split. Versions stay in one numbered sequence; each version's manifest records its scheme, and a scheme only appends to its own latest version. A scheme with no version yet gets a fresh one from ADR 0001's rule, with val and test sized `round(split_ratio × cameras)` instead of fixed at 1–2 cameras, so a larger `split_ratio` gives the larger held-out sets that ~18 cameras now allow. Versions written before this record no scheme and count as `identity` with the default eight behaviors.

## Considered Options

- **A script rewriting the exports**: a second copy of every snapshot, and nothing in the dataset key saying which grouping made it.
- **Remapping labels on the old split**: keeps test at two cameras chosen to cover eight behaviors, not three.
- **Replacing the old split**: loses comparability with the eight-behavior runs.

## Consequences

Datasets built before this have a different key and are not reused. Within a scheme a camera's split is still frozen; changing `split_ratio` only affects schemes that have no split yet. A classifier checkpoint trained on one scheme can't be evaluated against another's ground truth.
