"""src/utils/logger.py - Centralized logging with Rich formatting."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str, log_dir: str = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with Rich console output and file output."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    # Console handler with Rich
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
    )
    rich_handler.setLevel(level)
    
    # File handler
    log_file = Path(log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    
    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)
    
    return logger
