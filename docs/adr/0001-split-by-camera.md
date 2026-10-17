# Split train/val/test by camera, frozen and append-only

The models will be deployed on cameras that have no labels, and camera orientation is the hardest thing to generalize across, so every split is by camera: all of a camera's videos land in one split. Birds shared between camera folders (e.g. `H7`) are not grouped, because the views differ enough that the camera, not the bird, is what leaks. The split is chosen once by a deterministic rule (test is the camera pair that covers every behavior while removing the smallest share of each behavior's frames from training, val is the 1–2 remaining cameras chosen the same way covering as many behaviors as possible, the rest are train, which must keep every behavior), then frozen: newly labeled cameras are appended, existing cameras never move.

## Considered Options

- **Random per video** (`split_guidance_within_camera`): leaks near-identical footage from one camera across splits; kept only for the existing Label Studio pipelines.
- **Within-camera by recording time**: matches "same birds, more videos", but needs OCR of burned-in timestamps and doesn't test new cameras, which we will deploy on.

## Consequences

With 9 labeled cameras the test set is 1–2 cameras, so reported numbers come from 3-fold camera cross-validation (every camera tested once, same hyperparameters per fold). The frozen split is for day-to-day development and tuning only.
