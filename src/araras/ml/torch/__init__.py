"""PyTorch-specific helpers for training and Optuna integration."""

import importlib
import sys

__all__ = [
    "callbacks",
    "clear_torch_session",
    "EarlyStopping",
    "model",
    "optuna",
    "save_model_as_torchscript",
    "save_model_as_exported_program",
    "seed_everything",
    "TorchPruningCallback",
]

_attribute_map = {
    "callbacks": ("callbacks", None),
    "model": ("model", None),
    "optuna": ("optuna", None),
    "EarlyStopping": ("callbacks", "EarlyStopping"),
    "TorchPruningCallback": ("callbacks", "TorchPruningCallback"),
    "clear_torch_session": ("model", "clear_torch_session"),
    "save_model_as_torchscript": ("model", "save_model_as_torchscript"),
    "save_model_as_exported_program": ("model", "save_model_as_exported_program"),
    "seed_everything": ("model", "seed_everything"),
}


def _raise_optional_dependency_error(exc: ModuleNotFoundError) -> None:
    """Raise a consistent import error for missing torch dependencies.

    Args:
        exc (ModuleNotFoundError): Original module import error.

    Returns:
        None: This function always raises.

    Raises:
        ImportError: If torch-related modules are unavailable.
    """
    missing = getattr(exc, "name", "")
    if missing.startswith("torch") or missing.startswith("torchviz"):
        raise ImportError("Torch support requires the torch extra. Install with: pip install araras[torch]") from exc
    raise


def __getattr__(name: str):
    """Lazily resolve torch helpers and submodules.

    Args:
        name (str): Requested symbol name.

    Returns:
        Any: The requested submodule or symbol.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
        ImportError: If torch dependencies are missing.
    """
    if name not in _attribute_map:
        raise AttributeError(f"module {__name__!r} has no attribute {name}")

    module_name, symbol_name = _attribute_map[name]
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        _raise_optional_dependency_error(exc)

    value = module if symbol_name is None else getattr(module, symbol_name)
    setattr(sys.modules[__name__], name, value)
    return value


def __dir__() -> list[str]:
    """Return discoverable package attributes.

    Returns:
        list[str]: Sorted exported names.
    """
    return sorted(set(globals()) | set(__all__))
