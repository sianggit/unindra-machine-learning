"""Standardized logging utility for reproducible ML runs."""

import logging
import sys
from typing import Optional

def setup_logger(name: str = "ml-workspace", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured console logger.

    Args:
        name: Logger identifier name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
