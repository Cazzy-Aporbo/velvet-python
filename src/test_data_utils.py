"""
Tests for dataset and config utilities.
"""

from pathlib import Path
from your_package.data_utils import load_dataset, load_config


def test_load_dataset():
    dataset = load_dataset()
    # Should be a list of (text, label) pairs
    assert isinstance(dataset, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in dataset)
    # Ensure labels are not empty
    assert all(label for _, label in dataset)


def test_load_config(tmp_path: Path):
    # Copy config.yaml to a temp path to test loading
    config_src = Path("src/your_package/config.yaml")
    config_copy = tmp_path / "config.yaml"
    config_copy.write_text(config_src.read_text())

    config = load_config(config_copy)

    assert "model" in config
    assert config["model"]["type"] == "MockModel"
    assert config["training"]["epochs"] == 2