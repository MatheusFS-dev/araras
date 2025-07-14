"""
Last Edited: 14 July 2025
Description:
    Logging utilities and progress bar helpers.
"""
# ————————————————————————————— Standard Imports ————————————————————————————— #
import logging
import warnings
import traceback
from typing import *  # Wildcard import for adding type hints

# ————————————————————————————————— Constants ———————————————————————————————— #
# ANSI escape codes
RESET = "\x1b[0m"
WHITE = "\x1b[37m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
ORANGE = "\x1b[38;5;208m"
BLUE = "\x1b[94m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
BOLD = "\x1b[1m"


# ——————————————————————————— Logger configurations —————————————————————————— #
class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on the log level."""

    def __init__(self, fmt: str, datefmt: str | None = None) -> None:
        """Initialize the formatter with the given format strings.

        Args:
            fmt: Log message format string.
            datefmt: Optional date format string.
        """
        super().__init__(fmt, datefmt)
        self._colors = {
            logging.INFO: WHITE,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
            logging.CRITICAL: RED,
        }

    def format(self, record: logging.LogRecord) -> str:
        """Return the formatted record wrapped in the appropriate color codes."""
        text = super().format(record)
        color = self._colors.get(record.levelno, WHITE)
        return f"{color}{text}{RESET}"


def make_logger(
    name: str,
    fmt: str,
    datefmt: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger with its own StreamHandler and ColorFormatter.
    Propagation is turned off so it won’t inherit root handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # clear any existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(fmt, datefmt))
    logger.addHandler(handler)

    return logger


# 1) Simple Araras logger
logger = make_logger("araras", fmt="[ARARAS %(levelname)s] %(message)s")

# 2) Error-style logger with file/line info
logger_error = make_logger("araras_error", fmt="[%(pathname)s:%(lineno)d %(levelname)s] %(message)s")

# 3) Timestamp logger
logger_time = make_logger("araras_time", fmt="[%(asctime)s] %(levelname)s] %(message)s", datefmt="%H:%M:%S")

# —————————————————————————————————— Errors —————————————————————————————————— #
from rich.traceback import install

install(
    # show_locals=True,
)

# —————————————————————————————————— Checks —————————————————————————————————— #
from matplotlib import font_manager

if not any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
    logger.warning(
        f"{YELLOW}Times New Roman font not found. Install it by running {ORANGE}'sudo apt install msttcorefonts -qq && rm ~/.cache/matplotlib -rf'{YELLOW}."
    )

# ——————————————————————————————— Progress bar ——————————————————————————————— #
from tqdm import tqdm


def white_track(iterable, *, description: str, total: int):
    """Iterate with a white progress bar showing ``done/total``.

    Args:
        iterable (Iterable): The iterable to wrap and iterate over.
        description (str): A description to display alongside the progress bar.
        total (int): The total number of iterations.

    Yields:
        Any: Items from the provided iterable, while displaying the progress bar.
    """
    bar_fmt = "{percentage:3.0f}% {bar} {n_fmt}/{total_fmt} in {remaining}"
    with tqdm(
        iterable,
        total=total,
        desc=description,
        colour="white",
        ncols=100,
        bar_format=bar_fmt,
        unit="",  # remove default “it” label
    ) as pbar:
        for item in pbar:
            yield item


# ————————————————————————————— Supress warnings ————————————————————————————— #
from .ml.optuna.utils import supress_optuna_warnings

supress_optuna_warnings()
