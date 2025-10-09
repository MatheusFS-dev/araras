from typing import List, Optional

import os
import math
import time
import warnings
from pathlib import Path
from IPython.display import clear_output

from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()


def clear():
    """
    Clear all prints from terminal or notebook cell.

    This function works in multiple environments:
    - Jupyter notebooks/JupyterLab
    - Terminal/command prompt (Windows, macOS, Linux)
    - Python scripts run from command line
    """
    try:
        clear_output(wait=True)
    except:
        pass

    try:
        if os.name == "nt":  # Windows
            os.system("cls")
        else:  # macOS and Linux
            os.system("clear")
    except Exception as e:
        vp.printf(f"Error clearing terminal: {e}", tag="[ARARAS ERROR] ", color="red")


def format_number(number, precision=2):
    """Format ``number`` using metric suffixes.

    The value is scaled by powers of one thousand and annotated with the
    appropriate SI prefix (``K``, ``M`` and so on). Values below ``1`` use the
    milli/micro prefixes. Negative numbers retain their sign.

    Args:
        number (int | float): Value to format.
        precision (int): Number of decimal places to display. Defaults to ``2``.

    Returns:
        str: Formatted number with the corresponding prefix or ``"Invalid input: <number>"``.

    Raises:
        ValueError: If ``precision`` is negative.

    Notes:
        The value ``0`` is returned as ``"0"`` without a suffix.
    """

    original_value = number
    try:
        if precision < 0:
            raise ValueError("precision must be non-negative")

        if number == 0:
            return "0"

        is_negative = number < 0
        number = abs(number)

        suffixes = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
        small_suffixes = ["", "m", "μ", "n", "p", "f", "a", "z", "y"]

        if number < 1:
            suffix_index = 0
            while number < 1 and suffix_index < len(small_suffixes) - 1:
                number *= 1000
                suffix_index += 1
            formatted = f"{number:.{precision}f} {small_suffixes[suffix_index]}"
        else:
            suffix_index = 0
            while number >= 1000 and suffix_index < len(suffixes) - 1:
                number /= 1000
                suffix_index += 1
            formatted = f"{number:.{precision}f} {suffixes[suffix_index]}"

        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")

        return f"-{formatted}" if is_negative else formatted
    except Exception as e:
        vp.printf(f"Error formatting number: {e}", tag="[ARARAS ERROR] ", color="red")
        return f"Invalid input: {original_value}"


def format_bytes(bytes_value, precision=2):
    """Format a byte value with binary suffixes.

    The value is successively divided by ``1024`` and labelled with the
    appropriate binary prefix (``KB``, ``MB`` and so on). Negative inputs are
    supported and keep their sign.

    Args:
        bytes_value (int | float): Number of bytes to format.
        precision (int): Number of decimal places for the mantissa. Defaults to ``2``.

    Returns:
        str: Human readable string or ``"Invalid input: <bytes_value>"``.

    Raises:
        ValueError: If ``precision`` is negative.

    Warning:
        Inputs of ``0`` result in ``"0 B"``.
    """

    original_value = bytes_value
    try:
        if precision < 0:
            raise ValueError("precision must be non-negative")

        if bytes_value == 0:
            return "0 B"

        is_negative = bytes_value < 0
        bytes_value = abs(bytes_value)

        suffixes = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        suffix_index = 0

        while bytes_value >= 1024 and suffix_index < len(suffixes) - 1:
            bytes_value /= 1024
            suffix_index += 1

        formatted = f"{bytes_value:.{precision}f} {suffixes[suffix_index]}"

        if "." in formatted.split()[0]:
            number_part = formatted.split()[0].rstrip("0").rstrip(".")
            formatted = f"{number_part} {suffixes[suffix_index]}"

        return f"-{formatted}" if is_negative else formatted
    except Exception as e:
        vp.printf(f"Error formatting bytes: {e}", tag="[ARARAS ERROR] ", color="red")
        return f"Invalid input: {original_value}"


def format_scientific(number, max_precision=2):
    """Return ``number`` in scientific notation.

    The mantissa precision is automatically reduced to remove trailing zeros.
    Special floating point values such as ``nan`` and ``inf`` are returned
    unchanged. Negative values preserve their sign.

    Args:
        number (int | float): Value to format.
        max_precision (int): Maximum decimal places allowed in the mantissa. Defaults to ``2``.

    Returns:
        str: The formatted number or ``"Invalid input: <number>"``.

    Raises:
        ValueError: If ``max_precision`` is negative.
    """

    original_value = number
    try:
        if max_precision < 0:
            raise ValueError("max_precision must be non-negative")

        if number == 0:
            return "0"

        if math.isnan(number) or math.isinf(number):
            return str(number)

        exponent = math.floor(math.log10(abs(number)))
        mantissa = number / (10**exponent)

        if abs(mantissa) >= 10:
            mantissa /= 10
            exponent += 1

        precision = max_precision
        for p in range(max_precision + 1):
            test_mantissa = round(mantissa, p)
            if abs(test_mantissa - mantissa) < 1e-10:
                precision = p
                break

        mantissa_str = f"{mantissa:.{precision}f}".rstrip("0").rstrip(".")
        if exponent == 0:
            return mantissa_str
        return f"{mantissa_str}×10^{exponent}"
    except Exception as e:
        vp.printf(f"Error formatting scientific number: {e}", tag="[ARARAS ERROR] ", color="red")
        return f"Invalid input: {original_value}"


def format_number_commas(number, precision=2):
    """Format ``number`` with comma separators.

    Integers are displayed without a decimal part while floats include the
    specified precision. Invalid inputs result in an informative message.

    Args:
        number (int | float): Value to format.
        precision (int): Number of decimal places for floats. Defaults to ``2``.

    Returns:
        str: Number formatted with commas or ``"Invalid input: <number>"``.

    Raises:
        ValueError: If ``precision`` is negative.
    """

    original_value = number
    try:
        if precision < 0:
            raise ValueError("precision must be non-negative")

        if isinstance(number, int):
            return f"{number:,}"
        if isinstance(number, float):
            return f"{number:,.{precision}f}"

        raise TypeError(f"Invalid input type: {type(number)}. Must be int or float.")
    except Exception as e:
        vp.printf(f"Error formatting number with commas: {e}", tag="[ARARAS ERROR] ", color="red")
        return f"Invalid input: {original_value}"

# —————————————————————————————————— Others —————————————————————————————————— #
def supress_optuna_warnings() -> None:
    """Suppress Optuna experimental warnings.

    This helper inspects the Optuna package for the ``ExperimentalWarning``
    class in its possible locations and silences warnings triggered by
    experimental features.

    Notes:
        The :mod:`optuna` import occurs within this function to avoid pulling in
        optional dependencies at module import time.
    """

    import optuna

    warning_classes = []
    for module_name in ("_experimental", "exceptions"):
        module = getattr(optuna, module_name, None)
        if module is not None:
            warning_cls = getattr(module, "ExperimentalWarning", None)
            if warning_cls is not None:
                warning_classes.append(warning_cls)

    for cls in warning_classes:
        warnings.filterwarnings("ignore", category=cls)


# ——————————————————————————— Notebook Converter ———————————————————————————— #
class NotebookConverter:
    """Utility class for Jupyter notebook conversions."""

    @staticmethod
    def convert_notebook_to_python(
        notebook_path: Path,
        output_path: Optional[Path] = None,
        append_lines: Optional[List[str]] = None,
    ) -> Path:
        """Convert a Jupyter notebook to a Python script.

        By default, the script is written alongside ``notebook_path`` with a
        ``.py`` extension. Optionally, a custom ``output_path`` may be provided
        and additional ``append_lines`` can be written at the end of the
        generated file. This is useful when preparing notebooks for monitoring
        where extra code must be appended.

        Args:
            notebook_path (Path): Path to the ``.ipynb`` file.
            output_path (Optional[Path]): Optional custom destination for the converted script.
            append_lines (Optional[List[str]]): Optional lines to append to the resulting Python file.

        Returns:
            Path: Path to the created ``.py`` file.

        Raises:
            ImportError: If the ``nbformat`` dependency is missing.
            ValueError: If the notebook fails to convert or write to disk.
        """
        try:
            import nbformat
        except ImportError as e:
            raise ImportError(
                "Missing notebook dependencies. " "Please install: pip install nbformat"
            ) from e

        python_path = output_path or notebook_path.with_suffix(".py")

        try:
            # Load notebook with minimal memory footprint
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)

            # Extract code cells efficiently
            python_lines = [
                "#!/usr/bin/env python",
                "# -*- coding: utf-8 -*-",
                f"# Converted from: {notebook_path.name}",
                f'# Generated on: {time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())}',
                "",
            ]

            # Process cells with memory-efficient iteration
            for cell_idx, cell in enumerate(notebook.cells):
                if cell.cell_type == "code" and cell.source.strip():
                    # Add cell separator for debugging
                    python_lines.append(f"# Cell {cell_idx + 1}")

                    # Clean and add source code
                    source_lines = cell.source.strip().split("\n")
                    python_lines.extend(source_lines)
                    python_lines.append("")  # Empty line between cells

            if append_lines:
                python_lines.extend(append_lines)

            # Write to Python file atomically
            temp_path = python_path.with_suffix(".py.tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(python_lines))

                # Atomic rename for consistency
                temp_path.replace(python_path)

            except Exception:
                # Cleanup temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise

            return python_path

        except Exception as e:
            raise ValueError(f"Failed to convert notebook {notebook_path}: {e}") from e

    @staticmethod
    def convert_notebook_to_monitored_python(
        notebook_path: Path,
        success_flag: str,
    ) -> Path:
        """Convert a notebook to a monitored Python script.

        The resulting file is written next to ``notebook_path`` with the prefix
        ``temp_monitor_`` and includes a small snippet that writes the provided
        success flag when execution finishes.

        Args:
            notebook_path (Path): Path to the ``.ipynb`` file.
            success_flag (str): Path where the monitored script should write the
                ``SUCCESS`` flag upon completion.

        Returns:
            Path: Path to the generated monitored Python script.

        Raises:
            ImportError: If the ``nbformat`` dependency is missing.
            ValueError: If the notebook fails to convert or write to disk.
        """
        success_path = Path(success_flag).resolve()
        output_path = notebook_path.parent / f"temp_monitor_{notebook_path.stem}.py"
        append_lines = [
            "",  # newline before appended code
            "# Write success flag for the auto restart script",
            "from pathlib import Path",
            f"Path({repr(str(success_path))}).write_text('SUCCESS')",
            "",
        ]
        return NotebookConverter.convert_notebook_to_python(
            notebook_path,
            output_path=output_path,
            append_lines=append_lines,
        )
