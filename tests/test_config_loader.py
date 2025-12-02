"""Tests for configuration loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from utils.config_loader import ConfigLoader, load_config, load_training_config


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with test config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create base config
        base_config = {
            "project": {"name": "test_project", "seed": 42},
            "paths": {"data_root": "/data"},
        }
        with open(config_dir / "base_config.yaml", "w") as f:
            yaml.safe_dump(base_config, f)

        # Create model config
        model_config = {
            "model": {"type": "test_model", "size": "small"},
            "training": {"epochs": 100, "batch_size": 32},
        }
        with open(config_dir / "model.yaml", "w") as f:
            yaml.safe_dump(model_config, f)

        # Create data config
        data_config = {"data": {"train_split": 0.8, "val_split": 0.1}}
        with open(config_dir / "data.yaml", "w") as f:
            yaml.safe_dump(data_config, f)

        yield str(config_dir)


def test_config_loader_initialization(temp_config_dir):
    """Test ConfigLoader initialization."""
    loader = ConfigLoader(temp_config_dir)
    assert loader.config_dir.exists()


def test_config_loader_missing_dir():
    """Test ConfigLoader with missing directory."""
    with pytest.raises(FileNotFoundError):
        ConfigLoader("/nonexistent/directory")


def test_load_single_config(temp_config_dir):
    """Test loading a single configuration file."""
    loader = ConfigLoader(temp_config_dir)
    config = loader.load("model")

    assert "project" in config  # From base config
    assert "model" in config
    assert config["model"]["type"] == "test_model"
    assert config["project"]["seed"] == 42


def test_load_config_without_base(temp_config_dir):
    """Test loading config without merging base config."""
    loader = ConfigLoader(temp_config_dir)
    config = loader.load("model", base_config=None)

    assert "project" not in config
    assert "model" in config


def test_load_multiple_configs(temp_config_dir):
    """Test loading and merging multiple configs."""
    loader = ConfigLoader(temp_config_dir)
    config = loader.load_multiple("data", "model")

    assert "project" in config
    assert "model" in config
    assert "data" in config
    assert config["data"]["train_split"] == 0.8


def test_deep_merge():
    """Test deep dictionary merging."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    update = {"a": {"c": 99, "e": 4}, "f": 5}

    result = ConfigLoader._deep_merge(base, update)

    assert result["a"]["b"] == 1  # Preserved from base
    assert result["a"]["c"] == 99  # Updated
    assert result["a"]["e"] == 4  # Added
    assert result["d"] == 3  # Preserved
    assert result["f"] == 5  # Added


def test_save_config(temp_config_dir):
    """Test saving a configuration."""
    loader = ConfigLoader(temp_config_dir)
    config = {"test": {"value": 123}}

    output_path = Path(temp_config_dir) / "output.yaml"
    loader.save(config, str(output_path))

    assert output_path.exists()

    with open(output_path, "r") as f:
        loaded = yaml.safe_load(f)

    assert loaded["test"]["value"] == 123


def test_load_config_convenience_function(temp_config_dir):
    """Test convenience function for loading config."""
    config = load_config("model", config_dir=temp_config_dir)

    assert "model" in config
    assert "project" in config


def test_load_training_config_function(temp_config_dir):
    """Test convenience function for loading training config."""
    config = load_training_config("model", config_dir=temp_config_dir)

    assert "project" in config
    assert "data" in config
    assert "model" in config


def test_load_nonexistent_config(temp_config_dir):
    """Test loading a nonexistent config file."""
    loader = ConfigLoader(temp_config_dir)

    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")
