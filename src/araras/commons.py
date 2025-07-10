"""
Common setup for the Araras library.
This script is called in every other script within the Araras library.
"""

# ————————————————————————————— Standard Imports ————————————————————————————— #
import logging
import warnings
import traceback
from typing import * # Wildcard import for adding type hints

# ————————————————————————————————— Constants ———————————————————————————————— #
# ANSI escape codes
RESET = "\x1b[0m"
WHITE = "\x1b[37m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
ORANGE = "\x1b[38;5;208m"
BLUE = "\x1b[34m"


# ——————————————————————————— Logger configurations —————————————————————————— #
class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str | None = None):
        super().__init__(fmt, datefmt)
        self._colors = {
            logging.INFO: WHITE,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
            logging.CRITICAL: RED,
        }

    def format(self, record: logging.LogRecord) -> str:
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
logger = make_logger(
    "araras",
    fmt="[ARARAS %(levelname)s] %(message)s"
)

# 2) Error-style logger with file/line info
logger_error = make_logger(
    "araras_error",
    fmt="[%(pathname)s:%(lineno)d %(levelname)s] %(message)s"
)

# 3) Timestamp logger
logger_time = make_logger(
    "araras_time",
    fmt="[%(asctime)s] %(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# —————————————————————————————————— Checks —————————————————————————————————— #
try:
    import pretty_errors
except ImportError:
    logger.warning(
        "Module pretty_errors not found. Install it with 'pip install pretty_errors' for better error formatting."
    )

from matplotlib import font_manager
if not any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist):
    logger.warning(
        f"{YELLOW}Times New Roman font not found. Install it by running {ORANGE}'sudo apt install msttcorefonts -qq && rm ~/.cache/matplotlib -rf'{YELLOW}."
    )

# ————————————————————————————— Supress warnings ————————————————————————————— #
from araras.optuna.utils import supress_optuna_warnings
supress_optuna_warnings()
