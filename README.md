# bird_cv

Computer vision pipeline for detecting and classifying migratory songbird behavior from cage-mounted camera footage. Built around YOLOv11 for detection and VideoMAE for behavior classification.

## Overview

1. **Preprocessing** — Convert Label Studio annotations to YOLO format, run SAM2 cage segmentation, crop images per cage, and extract fixed-length behavior clips
2. **Detection** — Train a YOLO model to detect birds in cropped cage images
3. **Classification** — Fine-tune VideoMAE on labeled behavior clips (foraging, preening, resting, etc.)
4. **Evaluation** — Run YOLO tracking on the test set, compute MOT metrics, and evaluate VideoMAE classification

## Setup

```bash
make init
```

Requires `uv`. On the cluster, install the CUDA torch wheel after syncing:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

## Running Pipelines

Each pipeline reads from a `config.toml` in its directory. Set `run_id` to reuse an existing run directory, or leave it empty to auto-generate a timestamp.

```bash
python bird_cv/pipelines/preprocessing/pipeline.py
python bird_cv/pipelines/detection/pipeline.py
python bird_cv/pipelines/classification/pipeline.py
python bird_cv/pipelines/evaluation/pipeline.py
```

## Project Structure

```
bird_cv/
  classification/     — VideoMAE dataset, training, and evaluation
  detection/          — YOLO training, tracking, and MOT evaluation
  pipelines/          — End-to-end pipeline scripts and TOML configs
    preprocessing/
    detection/
    classification/
    evaluation/
  preprocessing/      — Label table generation, split guidance, YOLO cropping
  segmentation/       — SAM2 segmentation utilities
```

## Data Layout

Each pipeline run outputs to `{base_path}/{run_id}/`:

```
{run_id}/
  intermediate/       — Label tables, split guidance, clip/behavior index
  yolo_store/         — Full-frame YOLO images and labels
  yolo_crop/          — Per-cage cropped YOLO images and labels
  video_crop/         — Fixed-length behavior clips for VideoMAE
  segmentations/      — SAM2 cage mask JSONs
  training/           — Model checkpoints
  evaluation/         — Metrics, classification reports, confusion matrices
```