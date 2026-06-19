"""
This module provides a utility to load environment variables from a `.env` file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ev_at.core.logging import get_logger

__all__ = ["environment_loader"]

logger = get_logger("environment_loader")


class EnvironmentLoader:
    """
    Utility class to load environment variables from a .env file.
    This class follows the singleton pattern to ensure that environment variables
    are loaded only once during the application's lifetime.

    It checks for required environment variables and raises an error if any are missing.
    The required environment variables are defined in the _REQUIRED_ENV_VARS attribute.

    Example usage:

    .. code-block:: python

        from ev_at import environment_loader
        environment_loader.load_environment(dotenv_path=Path("/path/to/.env"))

    .. note::
        If the `dotenv_path` is not provided, it attempts to load the path from the
        `ENV_PATH` environment variable. If that variable is not set, it raises an error
        indicating that the path must be provided or set in the environment.
    """

    _REQUIRED_ENV_VARS = {"DATA_PATH"}

    def __new__(cls):
        """Create a singleton instance of EnvironmentLoader."""
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls._loaded = False
        return cls.instance

    def load_environment(self, dotenv_path: Path | None = None) -> None:
        """Loads environment variables from a .env file.

        Args:
            dotenv_path (Path, optional): Path to the .env file. If not provided,
                it attempts to load the path from the `ENV_PATH` environment variable.

        Raises:
            OSError: If the .env file cannot be found or if required environment
                variables are not set.
        """
        if self._loaded:
            logger.debug("Environment already loaded. Doing nothing.")
            return
        if dotenv_path is None:
            logger.debug(
                "No dotenv path provided. Attempting to load from environment "
                "variable 'ENV_PATH'."
            )
            try:
                dotenv_path = Path(os.environ["ENV_PATH"])
            except KeyError as e:
                msg = (
                    "Environment variable 'ENV_PATH' is not set. "
                    "Please set it to the path of your .env file "
                    "or provide the path directly."
                )
                logger.exception(msg)
                raise OSError(msg) from e

        logger.debug(f"Loading environment variables from {dotenv_path}")
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
        if not loaded:
            msg = (
                f"Failed to load environment variables from {dotenv_path}. "
                "Ensure the file exists and is readable."
            )
            logger.exception(msg)
            raise FileNotFoundError(msg)

        for var in self._REQUIRED_ENV_VARS:
            if not os.getenv(var):
                msg = (
                    f"Required environment variable '{var}' is not set. "
                    "The following environment variables are required: "
                    f"{', '.join(self._REQUIRED_ENV_VARS)}."
                )
                logger.exception(msg)
                raise OSError(
                    f"Required environment variable '{var}' is not set. "
                    "The following environment variables are required: "
                    f"{', '.join(self._REQUIRED_ENV_VARS)}. "
                )
        logger.info(f"Environment variables loaded successfully from {dotenv_path}")
        self._loaded = True


environment_loader = EnvironmentLoader()
