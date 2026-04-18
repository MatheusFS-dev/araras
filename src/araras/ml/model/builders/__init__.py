"""Model builder modules."""

import importlib
import sys

__all__ = ["cnn", "dnn", "gnn", "lm", "lstm", "se", "skip", "tcnn"]

_lazy_submodules = set(__all__)


def __getattr__(name: str):
	"""Lazily import builder submodules.

	Args:
		name (str): Requested builder module name.

	Returns:
		Any: Imported builder submodule.

	Raises:
		AttributeError: If ``name`` is not an exported builder.
		ImportError: If optional backend dependencies are missing.
	"""
	if name not in _lazy_submodules:
		raise AttributeError(f"module {__name__!r} has no attribute {name}")
	try:
		module = importlib.import_module(f".{name}", __name__)
	except ModuleNotFoundError as exc:
		missing = getattr(exc, "name", "")
		if missing.startswith("spektral"):
			raise ImportError("GNN builders require the gnn extra. Install with: pip install araras[gnn]") from exc
		raise
	setattr(sys.modules[__name__], name, module)
	return module


def __dir__() -> list[str]:
	"""Return discoverable attributes for the package.

	Returns:
		list[str]: Sorted exported names.
	"""
	return sorted(set(globals()) | set(__all__))
