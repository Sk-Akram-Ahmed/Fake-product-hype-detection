# ============================================================
# src/utils/logger.py
# Centralized logging using loguru
# ============================================================

import sys
from pathlib import Path
from loguru import logger as _logger


def get_logger(name: str = "hype_detection", log_dir: str = "logs") -> object:
    """
    Returns a configured loguru logger that writes to both
    the console (colored) and a rotating log file.

    Parameters
    ----------
    name    : str  — module name, used in log filename
    log_dir : str  — directory to store log files

    Returns
    -------
    loguru.Logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"

    # Remove default handler
    _logger.remove()

    # Console handler — colored, human-readable
    _logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level="DEBUG",
    )

    # File handler — plain text, rotates at 10 MB
    _logger.add(
        str(log_file),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        level="DEBUG",
        encoding="utf-8",
    )

    return _logger


# Module-level default logger
logger = get_logger("hype_detection")
