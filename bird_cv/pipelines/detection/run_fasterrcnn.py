from pathlib import Path
from bird_cv.detection.fasterrcnn_resnet import FasterRCNNConfig, load_and_train


cfg = FasterRCNNConfig(
    yolo_data_path=Path("/gscratch/pdoughe1/20260415_163412/yolo_crop"),
    output_path=Path("/gscratch/pdoughe1/20260415_163412/test_resnet")
)
load_and_train(cfg)

