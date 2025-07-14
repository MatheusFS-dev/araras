"""Top-level API for the araras package."""

from importlib.metadata import PackageNotFoundError, version

from .core import logger, logger_error, logger_time, white_track
from . import ml, notifications, runtime, utils, visualization

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # package not installed
    __version__ = "0.0.0"

__all__ = [
    "ml",
    "notifications",
    "runtime",
    "utils",
    "visualization",
    "logger",
    "logger_error",
    "logger_time",
    "white_track",
]
