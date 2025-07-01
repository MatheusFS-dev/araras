# tensoroo/__init__.py
"""Top-level API for the tensoroo package."""

import pkgutil
import importlib
import inspect

__all__ = []


for finder, module_name, ispkg in pkgutil.iter_modules(__path__):
    try:
        module = importlib.import_module(f'.{module_name}', __name__)
    except ImportError as e:
        continue

    for name, func in inspect.getmembers(module, inspect.isfunction):
        globals()[name] = func
        __all__.append(name)
