"""Optuna-related helpers for hyperparameter optimization."""

import importlib
import sys

__all__ = ["analyzer", "callbacks", "model_tools", "plots", "utils"]

_lazy_submodules = {"analyzer", "callbacks", "model_tools", "plots", "utils"}


def _raise_optional_dependency_error(exc: ModuleNotFoundError) -> None:
	"""Raise actionable dependency hints for optional backends.

	Args:
		exc (ModuleNotFoundError): Original import error.

	Returns:
		None: This function always raises.

	Raises:
		ImportError: With a backend-specific installation hint.
	"""
	missing = getattr(exc, "name", "")
	if missing.startswith("tensorflow"):
		raise ImportError(
			"TensorFlow-backed Optuna helpers require the tensorflow extra. "
			"Install with: pip install araras[tensorflow]"
		) from exc
	if missing.startswith("plotly"):
		raise ImportError(
			"Interactive plotting requires the viz extra. Install with: pip install araras[viz]"
		) from exc
	raise


def __getattr__(name: str):
	"""Lazily import optuna submodules.

	Args:
		name (str): Submodule name requested by the caller.

	Returns:
		Any: Imported submodule.

	Raises:
		AttributeError: If ``name`` is not exported.
		ImportError: If a required optional dependency is missing.
	"""
	if name not in _lazy_submodules:
		raise AttributeError(f"module {__name__!r} has no attribute {name}")
	try:
		module = importlib.import_module(f".{name}", __name__)
	except ModuleNotFoundError as exc:
		_raise_optional_dependency_error(exc)
	setattr(sys.modules[__name__], name, module)
	return module


def __dir__() -> list[str]:
	"""Return discoverable attributes for the package.

	Returns:
		list[str]: Sorted list of available attributes.
	"""
	return sorted(set(globals()) | set(__all__))
