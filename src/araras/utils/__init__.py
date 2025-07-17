"""Convenience imports for the ``araras.utils`` package.

This initializer exposes commonly used utilities while avoiding heavy
dependencies until their attributes are accessed. Functions from ``io`` and
``misc`` are lightweight, but ``system`` relies on TensorFlow.  To prevent
TensorFlow from loading when the :mod:`araras.utils` package is imported, the
attributes from ``system`` are loaded lazily via :func:`__getattr__`.

Notes:
    Importing any attribute listed in :data:`__all__` will trigger loading only
    the minimal submodule that defines it.  This keeps optional dependencies
    such as TensorFlow out of the import path unless explicitly requested.
"""

from __future__ import annotations

import importlib
import sys

__all__ = [
    "create_run_directory",
    "clear",
    "format_number",
    "format_bytes",
    "format_scientific",
    "format_number_commas",
    "NotebookConverter",
    "get_user_gpu_choice",
    "get_gpu_info",
    "gpu_summary",
    "log_resources",
]

_attribute_map = {
    "create_run_directory": ("io", "create_run_directory"),
    "clear": ("misc", "clear"),
    "format_number": ("misc", "format_number"),
    "format_bytes": ("misc", "format_bytes"),
    "format_scientific": ("misc", "format_scientific"),
    "format_number_commas": ("misc", "format_number_commas"),
    "NotebookConverter": ("misc", "NotebookConverter"),
    "get_user_gpu_choice": ("system", "get_user_gpu_choice"),
    "get_gpu_info": ("system", "get_gpu_info"),
    "gpu_summary": ("system", "gpu_summary"),
    "log_resources": ("system", "log_resources"),
}


def __getattr__(name: str):
    """Load attributes from submodules on demand.

    Args:
        name: Name of the attribute listed in :data:`__all__`.

    Returns:
        The requested attribute from its defining submodule.

    Raises:
        AttributeError: If ``name`` is not a known utility.
    """

    if name in _attribute_map:
        module_name, attr_name = _attribute_map[name]
        module = importlib.import_module(f".{module_name}", __name__)
        value = getattr(module, attr_name)
        setattr(sys.modules[__name__], name, value)
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name}")
