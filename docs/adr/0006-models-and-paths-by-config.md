# Models are picked by config kind; output paths live in one layout

Each pipeline role has one `kind` → class mapping, shared by training and inference: `[train.detector]` and `[train.classifier]` (`ModelSpec`) name a kind in `bird_cv/training/registry.py`, whose entry both trains that model and loads a checkpoint of it for inference (`load_for_inference`, built with the table's `infer_options`). The cage segmenter is never trained, so its mapping is `bird_cv/inference/registry.py`, picked by `[infer.segmenter] kind`. Pipelines never construct a model class; swapping a model is one registry entry plus a config change. Inference loads the checkpoints of a training run (`infer.py --train-run-id`), or a table's `checkpoint` for ad-hoc runs.

Every output path of the new pipelines (snapshots, splits, prepared ground truth and sampling, datasets per role and fold, training, inference and evaluation runs) is defined once in `bird_cv/core/layout.py`; config holds no output paths. Prepare ends by writing `prepared/current.json`, and later stages read their ground truth, split, CV folds and datasets through it.

## Considered Options

- **A separate inference registry for detectors and classifiers**: two mappings per role that could disagree on what `yolo` means.
- **Output paths as config fields** (as `[evaluate]` and `ModelSpec.dataset` first had): every stage had its own defaults and they didn't agree.

## Consequences

The legacy Label Studio pipelines keep their own config fields and paths. A rerun of prepare with new labels moves `current.json`, so stages run after it use the new data; each run's manifest records the snapshot and split it used.
