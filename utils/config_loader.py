"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Load and merge YAML configuration files."""

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

    def load(
        self, config_name: str, base_config: str = "base_config"
    ) -> Dict[str, Any]:
        """
        Load a configuration file, optionally merging with base config.

        Args:
            config_name: Name of config file (without .yaml extension)
            base_config: Name of base config to merge with (set to None to skip)

        Returns:
            Dictionary containing the merged configuration
        """
        config = {}

        # Load base config first
        if base_config is not None:
            base_path = self.config_dir / f"{base_config}.yaml"
            if base_path.exists():
                config = self._load_yaml(base_path)

        # Load and merge the specific config
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        specific_config = self._load_yaml(config_path)
        config = self._deep_merge(config, specific_config)

        return config

    def load_multiple(
        self, *config_names: str, base_config: str = "base_config"
    ) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files.

        Args:
            *config_names: Names of config files to load and merge
            base_config: Name of base config to merge with

        Returns:
            Dictionary containing the merged configuration
        """
        config = {}

        # Load base config
        if base_config is not None:
            base_path = self.config_dir / f"{base_config}.yaml"
            if base_path.exists():
                config = self._load_yaml(base_path)

        # Load and merge each config
        for config_name in config_names:
            config_path = self.config_dir / f"{config_name}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            specific_config = self._load_yaml(config_path)
            config = self._deep_merge(config, specific_config)

        return config

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            update: Dictionary to merge into base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def save(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Save a configuration dictionary to a YAML file.

        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(config_name: str, config_dir: str = "configs") -> Dict[str, Any]:
    """
    Convenience function to load a single configuration file.

    Args:
        config_name: Name of config file (without .yaml extension)
        config_dir: Directory containing configuration files

    Returns:
        Dictionary containing the configuration
    """
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)


def load_training_config(
    model_config: str, config_dir: str = "configs"
) -> Dict[str, Any]:
    """
    Load a complete training configuration (base + data + model).

    Args:
        model_config: Name of model config (e.g., 'board_detector', 'position_recognizer')
        config_dir: Directory containing configuration files

    Returns:
        Dictionary containing the merged configuration
    """
    loader = ConfigLoader(config_dir)
    return loader.load_multiple("data", model_config, base_config="base_config")
