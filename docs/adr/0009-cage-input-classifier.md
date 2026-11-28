# Classifier input is configurable; cage input skips detection

The bird-crop pipeline drops a behavior whenever YOLO fails to find the bird confidently: no box means no crop, so the frame is `no_bird`. To test whether removing that failure mode is worth losing the crop's resolution, `[models.classifier] options.input` picks what VideoMAE sees: `"bbox_crop"` (default, unchanged) or `"cage"`, whole cage frames padded black on the bottom and right to a square and resized to `options.input_size` px (default 224) in both training and inference. With `"cage"`, `ModelsConfig.roles()` leaves out the detector, so no stage trains, loads or runs YOLO; predictions have null boxes, presence metrics come back NaN and MOT is skipped, and the val sweep collapses to the `[infer]` `gap_fill_frames` / `det_conf`, which no longer affect anything.

Nothing predicts `no_bird` without a detector, so the cage classifier learns it as a class (amending ADR 0003 for this input only). `build_datasets` adds `no_bird` clips to prepare's behavior clips: back-to-back `num_frames` windows over unboxed stretches at least `num_frames` long (shorter gaps, widened, would be mostly bird), capped in train and val like resting at the split's most common active behavior clip count, seeded by `split_seed`. They are written in the dataset's `selection.parquet`. Prepare, the dataset key and the split are shared with the bird-crop input, so both train on the same dataset id and compare on the same test cameras; their training data live in `{role}/videomae/` and `{role}/videomae_cage/`. Inference classifies every non-empty cage frame with the same run windowing; frames where SAM2 finds no cage stay `no_bird`.

## Considered Options

- **A new kind, `videomae_cage`**: changing `kind` resets the spec to bare defaults, so swapping back and forth isn't one line.
- **`[models.detector] kind = "none"`**: the switch would sit on the role that doesn't decide what the classifier sees, and the two could disagree.
- **Never predicting `no_bird`**: every `no_bird` frame becomes a guaranteed error.
- **Keeping YOLO as a presence gate**: brings back the failure mode this removes.
- **Selecting `no_bird` clips in prepare**: changes every existing dataset key.

## Consequences

A cage checkpoint can't be used with `input = "bbox_crop"` or the reverse; inference refuses a detector with a cage classifier and the reverse. Small-bird behaviors may fall below what 224 px of a whole cage resolves; the comparison against bird crops (e2e macro-F1 and per-class recall, ADR 0005) is what decides.
