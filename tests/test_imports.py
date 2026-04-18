"""Import smoke tests for core and optional araras modules.

These tests validate that:
- Core package imports do not eagerly require heavyweight optional backends.
- Optional backend entrypoints behave predictably when dependencies are absent.

The suite is designed to run from the repository root before editable installation
by adding ``src`` to ``sys.path``.
"""

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Ensure the local ``src`` directory is importable.

    This allows running tests in a fresh virtual environment even before
    ``pip install -e .`` is executed.

    Returns:
        None: The function mutates ``sys.path`` in-place when needed.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _has_module(module_name: str) -> bool:
    """Return whether an importable module exists in the active environment.

    Args:
        module_name (str): Module to probe with ``importlib.util.find_spec``.

    Returns:
        bool: ``True`` when the module can be resolved, otherwise ``False``.
    """
    return importlib.util.find_spec(module_name) is not None


class TestImportBoundaries(unittest.TestCase):
    """Validate import behavior for core and optional modules."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare path state before running import smoke tests.

        Returns:
            None: Test class setup has no return value.
        """
        _ensure_src_on_path()

    def test_core_package_imports(self) -> None:
        """Core package modules should import without optional backends.

        Returns:
            None: Assertions validate import behavior.
        """
        araras = importlib.import_module("araras")
        self.assertIsNotNone(araras)

        ml = importlib.import_module("araras.ml")
        utils = importlib.import_module("araras.utils")

        # ``araras.visualization`` depends on matplotlib from core dependencies.
        if _has_module("matplotlib"):
            visualization = importlib.import_module("araras.visualization")
            self.assertIsNotNone(visualization)

        # ``araras.runtime`` depends on ``psutil`` from core package dependencies.
        # In a brand-new venv before ``pip install -e .``, this may be missing.
        if _has_module("psutil"):
            runtime = importlib.import_module("araras.runtime")
            self.assertIsNotNone(runtime)

        self.assertIsNotNone(ml)
        self.assertIsNotNone(utils)

    def test_lazy_ml_submodule_imports(self) -> None:
        """ML namespace imports should not eagerly load TensorFlow or Torch.

        Returns:
            None: Assertions validate lazy import boundaries.
        """
        model_pkg = importlib.import_module("araras.ml.model")
        optuna_pkg = importlib.import_module("araras.ml.optuna")
        torch_pkg = importlib.import_module("araras.ml.torch")

        self.assertIsNotNone(model_pkg)
        self.assertIsNotNone(optuna_pkg)
        self.assertIsNotNone(torch_pkg)

    def test_tensorflow_entrypoint_behavior(self) -> None:
        """TensorFlow-dependent entrypoints should be explicit about dependency state.

        When TensorFlow is present, fetching ``get_callbacks_model`` should work.
        When absent, requesting it should raise an import-related error.

        Returns:
            None: Assertions validate optional dependency behavior.
        """
        model_pkg = importlib.import_module("araras.ml.model")

        if _has_module("tensorflow"):
            callback_factory = getattr(model_pkg, "get_callbacks_model")
            self.assertTrue(callable(callback_factory))
        else:
            with self.assertRaises((ImportError, ModuleNotFoundError)):
                _ = getattr(model_pkg, "get_callbacks_model")

    def test_torch_entrypoint_behavior(self) -> None:
        """Torch-dependent entrypoints should map cleanly to optional deps.

        The torch callback class depends on both ``torch`` and ``optuna`` in the
        current implementation.

        Returns:
            None: Assertions validate optional dependency behavior.
        """
        torch_pkg = importlib.import_module("araras.ml.torch")

        if _has_module("torch") and _has_module("optuna"):
            early_stopping = getattr(torch_pkg, "EarlyStopping")
            self.assertTrue(callable(early_stopping))
        else:
            with self.assertRaises((ImportError, ModuleNotFoundError)):
                _ = getattr(torch_pkg, "EarlyStopping")


if __name__ == "__main__":
    unittest.main()
