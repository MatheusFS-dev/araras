"""High-level model utilities and builders."""

import importlib
import sys

__all__ = [
    "get_callbacks_model",
    "KParams",
    "convert_to_saved_model",
    "punish_model",
    "punish_model_flops",
    "punish_model_params",
    "print_tensor_mem",
    "validate_steps_per_execution",
    "get_flops",
    "get_macs",
    "get_inference_latency",
    "get_model_stats",
    "write_model_stats_to_file",
    "capture_model_summary",
    "run_dummy_inference",
    "builders",
]

_attribute_map = {
    "get_callbacks_model": ("callbacks", "get_callbacks_model"),
    "KParams": ("hyperparams", "KParams"),
    "convert_to_saved_model": ("tools", "convert_to_saved_model"),
    "punish_model": ("tools", "punish_model"),
    "punish_model_flops": ("tools", "punish_model_flops"),
    "punish_model_params": ("tools", "punish_model_params"),
    "print_tensor_mem": ("tools", "print_tensor_mem"),
    "validate_steps_per_execution": ("tools", "validate_steps_per_execution"),
    "get_flops": ("stats", "get_flops"),
    "get_macs": ("stats", "get_macs"),
    "get_model_stats": ("stats", "get_model_stats"),
    "get_inference_latency": ("stats", "get_inference_latency"),
    "write_model_stats_to_file": ("stats", "write_model_stats_to_file"),
    "capture_model_summary": ("utils", "capture_model_summary"),
    "run_dummy_inference": ("utils", "run_dummy_inference"),
    "builders": ("builders", None),
}


def _raise_optional_dependency_error(exc: ModuleNotFoundError) -> None:
    """Raise a user-facing error for missing optional backends.

    Args:
        exc (ModuleNotFoundError): Original import error raised while loading a module.

    Returns:
        None: This function always raises.

    Raises:
        ImportError: With an install hint for the required extra.
    """
    missing = getattr(exc, "name", "")
    if missing.startswith("tensorflow"):
        raise ImportError(
            "TensorFlow support requires the tensorflow extra. Install with: pip install araras[tensorflow]"
        ) from exc
    if missing.startswith("spektral"):
        raise ImportError("GNN support requires the gnn extra. Install with: pip install araras[gnn]") from exc
    raise


def __getattr__(name: str):
    """Lazily resolve model helpers and submodules.

    Args:
        name (str): Requested attribute name.

    Returns:
        Any: Requested function, class, or submodule.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
        ImportError: If a required optional dependency is missing.
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
    """Return discoverable attributes for the package.

    Returns:
        list[str]: Sorted attribute names exported by this package.
    """
    return sorted(set(globals()) | set(__all__))
