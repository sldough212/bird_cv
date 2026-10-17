# bird_cv

Turning cage-camera footage of migratory songbirds into frame-by-frame behavior labels for each bird.

## Media

**Camera**:
One fixed recording setup, identified by its folder name (e.g. `H7,I22`), which lists the birds in view. The unit of the train/val/test split.
_Avoid_: setup, rig, camera group

**Raw video**:
A video file as recorded, before standardization.

**Standardized video**:
A raw video re-encoded to 15 fps H.264. Its frames define the canonical frame index.

**Cage**:
One SAM2-segmented region of a camera's view holding at most one bird, identified by `cage_id`: its position in the camera's SAM2 prompt config.
_Avoid_: bird id (as an identifier), enclosure

**Cage video**:
A standardized video cropped and masked to one cage. The media labeled in the Detect project.

**Bird-crop video**:
A 224 px square video centered on the bird's box in a cage video, containing only frames that have a box. The media labeled in the Classify project.
_Avoid_: bbox video (in prose), crop video

## Frames and labels

**Canonical frame**:
A 0-indexed frame number of the standardized video. Every intermediate either uses it directly or carries a frame map to it.
_Avoid_: frame index (unqualified), Label Studio frame

**Frame map**:
The list giving, for each frame of a derived video, its canonical frame.

**Behavior**:
One of the eight labels in the Classify project (e.g. `head_move`, `resting`).

**Class**:
A behavior or `no_bird`; the nine values ground truth and predictions take.

**`no_bird`**:
The class for a canonical frame with no bird to classify: no Detect box in ground truth, no usable detection in predictions.
_Avoid_: no_detection, empty, background

**Labeling hole**:
A boxed frame with no behavior label. Excluded from scoring and counted.

**Labeled stretch**:
One contiguous labeled range of a single behavior on one cage.
_Avoid_: segment (ambiguous with SAM2 segments), behavior instance

## Data and runs

**Snapshot**:
An immutable copy of the Detect and Classify exports, named by a hash of their annotation content.

**Split version**:
A numbered, append-only assignment of cameras to train, val and test.

**Smoke test**:
A run that checks a pipeline executes and aligns frames correctly, using any model; its scores are never reported.
