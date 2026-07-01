"""
Logging Configuration for KLGrade API

Structured logging with rotation and formatting.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(
    name: str = "klgrade_api",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup structured logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory to store log files
        log_level: Logging level
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )

    # File handler with rotation
    log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_request(logger: logging.Logger, endpoint: str, filename: str, params: dict):
    """Log incoming API request."""
    logger.info(f"Request to {endpoint} | File: {filename} | Params: {params}")


def log_response(
    logger: logging.Logger, endpoint: str, status: str, processing_time: float
):
    """Log API response."""
    logger.info(
        f"Response from {endpoint} | Status: {status} | Time: {processing_time:.2f}ms"
    )


def log_error(logger: logging.Logger, endpoint: str, error: Exception):
    """Log API error."""
    logger.error(
        f"Error in {endpoint} | {type(error).__name__}: {str(error)}", exc_info=True
    )


def log_model_load(logger: logging.Logger, model_type: str, model_path: str):
    """Log model loading."""
    logger.info(f"Loading {model_type} model from {model_path}")


def log_inference(
    logger: logging.Logger, num_knees: int, num_js: int, num_ost: int, time_ms: float
):
    """Log inference metrics."""
    logger.info(
        f"Inference complete | Knees: {num_knees} | JS lesions: {num_js} | "
        f"OST lesions: {num_ost} | Time: {time_ms:.2f}ms"
    )


# Create default logger instance
api_logger = setup_logger()
