"""Structured logging via loguru. Import `logger` from here everywhere."""

import sys

from loguru import logger

from app.config import PROJECT_ROOT, settings

# Remove default handler
logger.remove()

# Console output — human-readable
logger.add(
    sys.stdout,
    level=settings.log_level,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    ),
    colorize=True,
)

# File output — JSON for structured analysis
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "app.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="30 days",
    compression="gz",
)

__all__ = ["logger"]
