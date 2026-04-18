"""Top-level API for the araras package."""

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
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

_lazy_submodules = {"ml", "notifications", "runtime", "utils", "visualization"}


def __getattr__(name: str):
    """Lazily import optional subpackages.

    Args:
        name (str): Name of a subpackage listed in ``__all__``.

    Returns:
        The requested module once imported.

    Raises:
        AttributeError: If ``name`` is not a known subpackage.
    """
    if name in _lazy_submodules:
        module = importlib.import_module(f".{name}", __name__)
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
