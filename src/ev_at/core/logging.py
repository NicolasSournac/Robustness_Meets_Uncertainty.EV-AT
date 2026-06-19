"""
Project-wide logging configuration using loguru.

This module provides a centralized logging configuration that can be imported
and used across all modules in the project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from loguru._logger import Logger

__all__ = [
    "get_logger",
    "set_log_level",
    "add_handler",
    "remove_handler",
    "default_logger",
]


class EVATLogger:
    """
    Centralized logger configuration for the ev_at project.

    This class provides a singleton pattern to ensure consistent logging
    configuration across the entire application.
    """

    _instance: EVATLogger | None = None
    _initialized: bool = False
    _VALID_LOG_LEVELS = {
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    def __new__(cls) -> EVATLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._setup_logger()
            EVATLogger._initialized = True

    def _setup_logger(self) -> None:
        """Configure the logger with appropriate handlers and formatting."""

        logger.remove()
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in EVATLogger._VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log level: `{log_level}`. "
                "Check your environment variables or configuration. "
                f"Valid levels are: {', '.join(EVATLogger._VALID_LOG_LEVELS)}."
            )

        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<magenta>{extra[module]}</magenta> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        log_path = Path(
            os.getenv("LOG_PATH", Path.home() / "workspace" / "logs" / "ev_at")
        )

        log_path.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{extra[module]} | "
            "{message}"
        )

        logger.add(
            log_path / "ev_at.log",
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

        logger.add(
            log_path / "ev_at_errors.log",
            format=file_format,
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

        logger.configure(
            extra={
                "project": "ev_at",
            }
        )

    def get_logger(self, name: str | None = None) -> Logger:
        """
        Get a logger instance with optional name binding.

        Args:
            name: Optional name to bind to the logger for better identification

        Returns:
            Logger instance bound with the specified name
        """
        if name:
            return logger.bind(module=name)  # type: ignore
        return logger  # type: ignore

    def set_level(self, level: str) -> None:
        """
        Dynamically change the logging level.

        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """

        logger.remove()
        os.environ["LOG_LEVEL"] = level.upper()
        self._setup_logger()

    def add_custom_handler(
        self,
        sink: Any,
        format_string: str | None = None,
        level: str = "INFO",
        **kwargs: Any,
    ) -> int:
        """
        Add a custom log handler.

        Args:
            sink: Where to send the logs (file path, stream, etc.)
            format_string: Custom format string for this handler
            level: Minimum level for this handler
            **kwargs: Additional arguments for the handler

        Returns:
            Handler ID that can be used to remove the handler later
        """
        if format_string is None:
            format_string = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )

        return logger.add(sink, format=format_string, level=level, **kwargs)

    def remove_handler(self, handler_id: int) -> None:
        """
        Remove a log handler by its ID.

        Args:
            handler_id: ID of the handler to remove
        """
        logger.remove(handler_id)


_ev_at_logger = EVATLogger()

get_logger = _ev_at_logger.get_logger
set_log_level = _ev_at_logger.set_level
add_handler = _ev_at_logger.add_custom_handler
remove_handler = _ev_at_logger.remove_handler
default_logger = _ev_at_logger.get_logger("ev_at")
