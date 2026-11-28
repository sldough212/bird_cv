# bird_cv

Computer vision pipeline for detecting and classifying migratory songbird
behavior in cage-mounted camera footage. Each raw video is standardized to a
fixed frame rate, split into cages (SAM2), searched for birds (YOLO or RF-DETR) and
classified per frame (VideoMAE), then scored against Label Studio
annotations.

## Setup

```bash
make init
```

Requires `uv`. On Linux, `uv sync` installs the CUDA 12.4 torch wheels.

## Running the pipeline

Everything reads one config, `bird_cv/pipelines/configs/config.toml`. On the
cluster, submit the whole chain as SLURM jobs:

```bash
./slurm_scripts/submit.sh            # full run
./slurm_scripts/submit.sh --smoke    # 2 epochs, 2 videos per inference run
```

`submit.sh` chains these stages, each one also runnable by itself as
`python bird_cv/pipelines/<stage>.py CONFIG ...`:

1. **prepare**: snapshot the Label Studio exports, relabel the Classify
   labels (`[prepare] relabel`), build ground truth on canonical frames,
   assign cameras to the label scheme's frozen split, sample training frames
   and clips.
2. **build_datasets**: write the YOLO and VideoMAE training data.
3. **train** (one job per `[models]` role) and its model evaluation.
4. **infer**: run end to end on the val cameras for every `[evaluate]`
   (gap fill, det_conf) setting, then on the test cameras at the best one.
5. **evaluate**: pick the best sweep setting, then score the test run.

Useful options:

- `RUN_ID=name`: the run's id (default: a timestamp). Every stage writes under it.
- `--dataset ID` (or `DATASET=ID`): skip prepare and train on an existing dataset.
- `--detect-export F --classify-export F`: use existing exports instead of
  pulling them from Label Studio.

Models are picked in config: `[models.detector]` and `[models.classifier]`
`kind`, and `[infer.segmenter]`. Swap detectors with one word,
`[models.detector] kind = "yolo"` or `"rfdetr"`; each kind's settings live in
`[models.detector.<kind>]`. Ray Tune is off unless a model sets
`tune = true`. `standard_fps` sets the frame rate. `[prepare] relabel` maps
the Classify labels onto a label scheme whose classes are the class vocabulary
(`config.toml` groups them into awake / restful / restless). To
relabel or resplit already downloaded data, rerun prepare with
`--detect-export` / `--classify-export` pointing at a dataset's
`annotations/` files.

`[models.classifier] options.input = "cage"` classifies whole cage frames
instead of bird crops and drops the detector from every stage (ADR 0009);
remove it to go back. It trains on the same dataset id, so the two runs
compare directly.

### Label Studio pipelines

`upload_for_detect`, `upload_for_classify`, `upload_for_retrain_detect`,
`retrain_classify` and `detect` create pre-labels and retrain models
for Label Studio. They have their own `slurm_scripts/run_*.sh` and still use
the older paths derived from `path_to_base_output`.

## Output layout

Under `path_to_base_output`:

```
splits/split_v{n}.parquet        frozen camera split of one label scheme
datasets/{id}/                   prepare + build_datasets output; a symlink if
                                 an identical dataset already exists
models/{role}/{id}/              a training run
infer/{id}/val_sweep/gap*_conf*/ val inference per sweep setting
infer/{id}/test/                 test inference at the chosen setting
eval/{id}/{sweep,test,detector,classifier}/   metrics
slurm_logs/{id}/                 job logs
```

See `bird_cv/core/layout.py` for every file.

## Project structure

```
bird_cv/
  core/            layout, manifests, ids, frames, class vocabulary, schemas
  preprocessing/   snapshots, ground truth, splits, sampling, standardize, crop
  training/        model registry and training entry points
  detection/       YOLO and RF-DETR backends
  classification/  VideoMAE backend
  segmentation/    SAM2 cage segmentation
  inference/       end-to-end inference on raw videos
  evaluation/      all metrics and plots
  client/          Label Studio client
  pipelines/       entry points and config
slurm_scripts/     SLURM job scripts and submit.sh
docs/adr/          design decisions
```

## Development

```bash
uv run pre-commit run --all-files   # ruff, ruff-format, mypy
uv run pytest tests
```
