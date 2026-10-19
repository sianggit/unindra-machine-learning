"""Configuration loader utility."""

from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file safely.

    Args:
        config_path: Relative or absolute path to the YAML file.

    Returns:
        Dictionary containing configuration properties.

    Raises:
        FileNotFoundError: If configuration file is missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}
