"""Machine learning utilities and tools."""

import importlib
import sys

__all__ = ["model", "optuna", "torch"]

_lazy_submodules = {"model", "optuna", "torch"}


def __getattr__(name: str):
	"""Lazily import ML subpackages.

	Args:
		name (str): Requested attribute name.

	Returns:
		Any: Imported submodule object.

	Raises:
		AttributeError: If ``name`` is not a known submodule.
	"""
	if name in _lazy_submodules:
		module = importlib.import_module(f".{name}", __name__)
		setattr(sys.modules[__name__], name, module)
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name}")


def __dir__() -> list[str]:
	"""Return discoverable attributes for the package.

	Returns:
		list[str]: Sorted list of module attributes exposed by this package.
	"""
	return sorted(set(globals()) | set(__all__))
