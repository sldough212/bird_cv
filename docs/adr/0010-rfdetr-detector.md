# RF-DETR is a second detector kind, tracked like YOLO

To compare a transformer detector against YOLO, `[models.detector] kind = "rfdetr"` trains and runs Roboflow's RF-DETR (`rfdetr`, Apache-2.0 sizes only: `pretrained` names the class, `RFDETRSmall` by default). Swapping is one word: each kind's settings live in a `[models.detector.<kind>]` subtable, merged onto the role's shared keys, and every subtable is validated whether active or not.

RF-DETR trains on the `yolo` kind's dataset in place: trainers declare a `data_format`, kinds sharing one share `datasets/{id}/{role}/{data_format}/`, and RF-DETR's training writes a `data.yaml` into it. Nothing is prepared or built again, so both detectors train on the same frames. Within a run RF-DETR keeps its best epoch by EMA val mAP (`checkpoint_best_total.pth`); the checkpoint is then scored on val presence F1 with the same evaluator as YOLO (ADR 0005). It has no Ray Tune integration, so `tune = true` logs a warning and trains once.

Inference feeds RF-DETR's top box per frame into the same Ultralytics tracker and `tracker.yaml` as YOLO, one tracker per cage. YOLO's BoT-SORT does more than assign ids: it outputs Kalman-smoothed boxes, waits for new tracks to be confirmed and gates jumps, and untracked boxes are dropped. Sharing it keeps the detector the only difference between the two runs. RF-DETR's `threshold` stays under `track_low_thresh` so the tracker's second pass gets boxes.

## Considered Options

- **No tracker** (top box per frame): RF-DETR would lose the confirmation and smoothing YOLO gets, mixing tracker and detector effects.
- **Roboflow's `trackers` BoT-SORT**: a public API, but it returns raw boxes and scales its buffer by fps, so it would track differently from YOLO.
- **A separate RF-DETR dataset**: duplicates every frame for an identical YOLO-format layout.

## Consequences

The tracker is driven through Ultralytics internals (`TRACKER_MAP`, `Boxes`), pinned by `uv.lock`; `tests/detection/test_rf_detr.py` breaks if they change. `rfdetr[train]` pins the pre-release `torch-hungarian==0.1.0rc0`, allowed by a uv constraint.
