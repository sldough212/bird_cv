# Models are picked by config kind; output paths live in one layout

Each pipeline role has one `kind` → class mapping, shared by training and inference: `[models.detector]` and `[models.classifier]` (`ModelSpec`) name a kind in `bird_cv/training/registry.py`, whose entry both trains that model and loads a checkpoint of it for inference (`load_for_inference`, built with the table's `infer_options`). The cage segmenter is never trained, so its mapping is `bird_cv/inference/registry.py`, picked by `[infer.segmenter] kind`. `[train]` holds only training-run settings. Pipelines never construct a model class; swapping a model is one registry entry plus a config change. Inference loads the checkpoints of a training run (`infer.py --train-run-id`), or a table's `checkpoint` for ad-hoc runs. The classifier's `options.num_frames` is the only clip length: prepare samples clips of that length.

Every output path of the new pipelines (snapshots, splits, prepared ground truth and sampling, datasets per role, kind and fold, training, inference and evaluation runs, and the result files inside them) is defined once in `bird_cv/core/layout.py`; config holds no output paths. Prepare prints a prepared-data id (`Prepared.id`, the sampling directory relative to `prepared/`), and train and evaluate take it as `--prepared`; `submit.sh` captures it once and passes it to every job. `prepared/current.json` is only a fallback, logged as a warning. Every train, infer and evaluate manifest records the prepared data, and evaluate refuses to score runs that recorded different data.

## Considered Options

- **A separate inference registry for detectors and classifiers**: two mappings per role that could disagree on what `yolo` means.
- **Output paths as config fields** (as `[evaluate]` and `ModelSpec.dataset` first had): every stage had its own defaults and they didn't agree.
- **Always reading the latest prepare output**: a rerun of prepare between jobs would silently mix data.

## Consequences

The legacy Label Studio pipelines keep their own config fields (`[detect]`, `[classify]`) and paths, so a few values (checkpoints, tracker) are duplicated in `[models.*]`.
