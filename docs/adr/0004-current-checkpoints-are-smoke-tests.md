# Existing checkpoints are for smoke tests only

The YOLO and VideoMAE checkpoints that exist before the frozen split were trained on splits that leaked across cameras and perform poorly. They are used only to check that pipelines run and produce correctly aligned output; no metric computed with them is reported or used as a baseline. All new training starts from generic pretrained weights (`yolo11n.pt`, `MCG-NJU/videomae-base`), never a previous fine-tuned checkpoint, so every model is reproducible from its snapshot and split version alone.
