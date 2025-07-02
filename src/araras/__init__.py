# araras/__init__.py
"""Top-level API for the araras package."""

import pkgutil
import importlib
import inspect
from importlib.metadata import version, PackageNotFoundError

__all__ = []

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # package not installed
    print(f"Package '{__name__}' is not installed.")
    __version__ = "0.0.0"


for finder, module_name, ispkg in pkgutil.iter_modules(__path__):
    try:
        module = importlib.import_module(f'.{module_name}', __name__)
    except ImportError as e:
        continue

    for name, func in inspect.getmembers(module, inspect.isfunction):
        globals()[name] = func
        __all__.append(name)
