from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data_utils import load_config


def test_load_config_raises_when_file_does_not_exist(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_config_rejects_non_mapping_payload(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level dict"):
        load_config(cfg)


def test_load_config_rejects_scalar_payload(tmp_path: Path) -> None:
    cfg = tmp_path / "scalar.yaml"
    cfg.write_text("123", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level dict"):
        load_config(cfg)


def test_load_config_raises_on_invalid_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "invalid.yaml"
    cfg.write_text("model: [unclosed", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_config(cfg)


def test_load_config_returns_empty_dict_for_empty_file(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("", encoding="utf-8")

    assert load_config(cfg) == {}
