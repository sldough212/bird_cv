from pathlib import Path
from types import SimpleNamespace

import yaml

from bird_cv.detection.train_yolo import tuned_hyperparameters


def test_tuned_hyperparameters_come_from_the_run_args(tmp_path: Path) -> None:
    run = tmp_path / "bird_yolo6"
    run.mkdir()
    (run / "args.yaml").write_text(
        yaml.safe_dump({"lr0": 0.01, "box": 7.5, "epochs": 60})
    )
    results = [
        SimpleNamespace(config={"lr0": 0.002, "box": 9.0, "data": "x.yaml"}),
        SimpleNamespace(config={"lr0": 0.003, "box": 1.0, "data": "x.yaml"}),
    ]
    assert tuned_hyperparameters(results, run) == {"box": 7.5, "lr0": 0.01}  # type: ignore[arg-type]
