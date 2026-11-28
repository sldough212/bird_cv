# V-JEPA 2 is a second classifier kind; bird crops other than 224 px are recropped

To compare a second video backbone against VideoMAE by changing only config, `[models.classifier] kind = "vjepa2"` fine-tunes Meta's V-JEPA 2 (`VJEPA2ForVideoClassification` in transformers; generic pretrained `facebook/vjepa2-vitl-fpc64-256` per ADR 0004). Training, tuning, evaluation and windowed inference are shared with VideoMAE; a `Backbone` in `bird_cv/classification/backbones.py` names only what differs (model class, processor, head modules, input-size config attribute), and `VJEPA2Trainer` subclasses `VideoMAETrainer`. Both runs use the same prepare selection, split and 16-frame clips, so runs on one dataset id compare directly.

`options.input_size` is the classifier's input side for both inputs. Bird crops at 224 px are the Classify bird-crop videos, as before. At any other size, build_datasets recrops them from the cage videos: each cage's boxed frames, cut at `input_size` around the ground-truth box centers smoothed as inference smooths them, so the crop has more context at the same pixel scale and is never resized. The recrop keeps a bird-crop video's indexing, so clip ranges are unchanged, and inference crops at the same size. `[infer] crop_size` is gone. The dataset directory adds the size when it isn't 224 (`classifier/vjepa2_256/`), so sizes don't overwrite each other under one dataset id.

V-JEPA 2's RoPE positions follow `config.crop_size`, so training sets it to `input_size` and inference refuses a checkpoint whose `crop_size` differs; VideoMAE's position embeddings fix it to 224. Config load checks that a `vjepa2` `input_size` is a multiple of 16, `num_frames` is even and `pretrained` is a `facebook/vjepa2-*` model or a local path, and that `videomae` uses 224. Cage input works for both kinds.

With `freeze_encoder = true` (the default), V-JEPA 2 trains its attentive pooler and linear classifier, which are new: a frozen random pooler would mix features arbitrarily. VideoMAE's head is mean pooling plus `fc_norm` and a linear classifier, of which only the linear layer trains, as before.

## Considered Options

- **One generic `hf_video` kind picked by `pretrained`**: moves VideoMAE too, for no gain while there are two models.
- **Separate `vjepa2_*` modules**: four near-copies of the VideoMAE code.
- **Sharing the 224 px clips and resizing to 256**: upsamples the crops and adds no information.
- **Fetching the checkpoint's config at config load**: puts network access in every stage; checkpoint configs are checked when the model loads instead.

## Consequences

The 16-frame, frozen-encoder setup is a baseline, not V-JEPA 2's best. Ways to improve training later, each a config or small trainer change: unfreeze the last few encoder blocks with layer-wise learning-rate decay, train in bf16 to afford full fine-tuning or larger batches, use longer clips (the checkpoint was pretrained on 64 frames; needs a new prepare), try larger checkpoints (`vith`, `vitg`, `vitg-fpc64-384`), and add the pooler's learning rate as its own search-space parameter.
