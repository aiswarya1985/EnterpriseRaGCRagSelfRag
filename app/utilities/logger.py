# logger_config.py
import os
import sys
from loguru import logger

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - {message}"
)

def setup_logger():
    """Call this ONCE at entry points (main.py / app.py)."""
    os.makedirs("logs", exist_ok=True)
    logger.remove()  # Clear default handlers

    # Console output
    logger.add(sys.stderr, format=LOG_FORMAT, level="INFO")

    # Single centralized log file
    logger.add(
        "logs/logs_rag.txt",
        format=LOG_FORMAT,
        level="INFO",
        colorize=False,
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8"
    )