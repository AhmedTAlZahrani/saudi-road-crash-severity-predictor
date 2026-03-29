"""Logging configuration for the crash severity predictor.

Sets up a RotatingFileHandler for persistent logs and a console handler
for real-time feedback during training runs.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level=logging.INFO):
    """Configure the 'crash_predictor' logger with rotating file and console output.

    Args:
        level: Minimum log level. Defaults to INFO.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("crash_predictor")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_dir / "training.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
