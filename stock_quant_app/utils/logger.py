"""Structured logging via loguru. Import `logger` from here everywhere."""

import sys

from loguru import logger

from app.config import settings

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

# File output — structured JSON lines by default (serialize=True puts
# logger.bind(...) fields into the "extra" object of each record)
LOG_DIR = settings.log_dir
LOG_DIR.mkdir(parents=True, exist_ok=True)

if settings.log_json:
    logger.add(
        LOG_DIR / "app.jsonl",
        level="DEBUG",
        serialize=True,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )
else:
    logger.add(
        LOG_DIR / "app.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
    )

__all__ = ["logger"]
