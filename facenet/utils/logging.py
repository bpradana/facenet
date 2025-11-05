"""Logging utilities for the FaceNet project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    output_dir: str,
    *,
    log_level: int = logging.INFO,
    filename: str = "training.log",
) -> Path:
    """Configure root logger with console and file handlers.

    Returns the path to the log file.
    """

    log_path = Path(output_dir) / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicating handlers when reconfiguring.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    return log_path


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger."""

    return logging.getLogger(name or "facenet")
