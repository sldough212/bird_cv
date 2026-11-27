# A trained cage segmenter can replace the one-shot SAM2 prompts

The `sam2` segmenter needs a hand-made prompt config and reference frame per camera, so the pipeline can't run on a camera nobody has configured. The `yolo_cages` segmenter (`[infer.segmenter] kind = "yolo_cages"`) uses a model trained to find cages instead, so it needs no per-camera setup; switching between the two is a config change. Its model is the opt-in `[models.cage_detector]` role (kind `yolo_seg`, an Ultralytics segmentation model with one `cage` class), which prepare, build_datasets, train and evaluate handle like the other roles. It trains on the legacy one-shot SAM2 masks (`run_segment` output, `[prepare] path_to_cage_masks`), taken as ground truth, on frames decoded from the raw videos (legacy segment frame `i` is raw frame `i + 1`), and is selected on val mask mAP50-95.

Cages never move within a video, so the model runs on only a few evenly spaced frames per video (`sample_frames`). Detections are matched across those frames on IoU, each cage gets one static mask (a pixel-wise majority vote) for the whole video, and cages are numbered row-major. Every video of a camera then gets the same ids as long as the camera isn't re-aimed; on new cameras a cage id is a position, not a bird. `expected_cages` (per camera or for all) keeps the most confident cages and fails a video that has fewer; it defaults to automatic.

Inference saves every video's keyframe masks (`segments/`), whatever the segmenter. `evaluate segments` compares a test run's masks with the legacy masks of the test cameras (matched IoU, worst cage, count accuracy), and two `submit.sh` runs that differ only in segmenter kind compare end to end.

## Considered Options

- **A cage box detector prompting SAM2**: this was the first plan. A segmentation model gives the masks directly, and with static cages SAM2's propagation adds nothing.
- **Fine-tuning SAM2 with automatic prompts**: harder to train, and the filtering of masks needs tuning.
- **A discovery job per camera before inference**: detecting on each video is cheap, and row-major ids already agree across a camera's videos.

## Consequences

Masks predicted at low resolution and upsampled may be coarser at cage edges than SAM2's; `evaluate segments` measures this. SAM refinement can be added if it matters. The legacy Label Studio path stays on one-shot SAM2.
