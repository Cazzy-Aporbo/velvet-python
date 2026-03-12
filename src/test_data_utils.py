from pathlib import Path
from src.data_utils import load_dataset, load_config


def test_load_dataset():
    dataset = load_dataset()
    assert isinstance(dataset, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in dataset)
    assert all(label for _, label in dataset)


def test_load_config(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  type: MockModel\ntraining:\n  epochs: 2\n")
    config = load_config(cfg)
    assert config["model"]["type"] == "MockModel"
    assert config["training"]["epochs"] == 2