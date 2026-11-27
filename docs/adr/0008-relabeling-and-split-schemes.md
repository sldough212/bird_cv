# Relabel before ground truth; one split lineage per label scheme

Eight behaviors are too many and too rare for VideoMAE to learn from our label counts, and labelers disagree on the fine distinctions. Prepare therefore runs a **relabeler** (`[prepare] relabel`, a `Relabeler` class in `bird_cv/preprocessing/relabel.py` whose fields are `relabel_options`) on the Classify export before ground truth. Its `behaviors` are the class vocabulary, so everything downstream (split, sampling, training, evaluation) sees only the relabeled classes and config doesn't repeat them; an unknown name or option fails when the config loads. `identity` keeps the Classify labels; `activity_groups` maps them to `restful` (resting, and each drawn `head_move` span of at most `head_move_restful_max_frames` = 15 bird-crop frames), `awake` (longer head moves, preening, fluffing, tail wag) and `restless` (flying, walking/hopping, flapping wings). The snapshot stays as exported, labelers keep labeling the fine classes, and the Label Studio pipelines are unchanged.

A **label scheme** (relabeler plus the behavior names it outputs) changes which cameras hold which classes, so it gets its own split. Versions stay in one numbered sequence; each version's manifest records its scheme, and a scheme only appends to its own latest version. A scheme with no version yet gets a fresh one from ADR 0001's rule, changed in two ways. Val and test are sized `round(split_ratio × cameras)` instead of fixed at 1–2 cameras. And they are the cameras whose share of each behavior's frames is closest to their `split_ratio` (lowest sum of squared differences), not the ones taking the fewest frames from train: with more held-out cameras, the old objective filled them with the cameras that had the least of the rare classes (on the 17 grouped cameras, 146 restless and 89 awake test frames, each restless split from one cage), leaving nothing to evaluate. Coverage stays a hard rule for test and train and best effort for val. Versions written before this record no scheme and count as `identity` with the default eight behaviors.

## Considered Options

- **A script rewriting the exports**: a second copy of every snapshot, and nothing in the dataset key saying which grouping made it.
- **Remapping labels on the old split**: keeps test at two cameras chosen to cover eight behaviors, not three.
- **Replacing the old split**: loses comparability with the eight-behavior runs.
- **Keeping the fewest-frames objective**: protects train, but leaves held-out rare classes on one camera.

## Consequences

Datasets built before this have a different key and are not reused. Within a scheme a camera's split is still frozen; changing `split_ratio` only affects schemes that have no split yet. A classifier checkpoint trained on one scheme can't be evaluated against another's ground truth.
