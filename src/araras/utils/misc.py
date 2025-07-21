from araras.core import *

import os
import math
import time
from pathlib import Path
from typing import List, Optional
from IPython.display import clear_output


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
        logger_error.error(f"{RED}Error clearing terminal: {e}{RESET}")


def format_number(number, precision=2):
    """
    Format a number using scientific suffixes.

    Args:
        number (int, float): The number to format
        precision (int): Number of decimal places to show (default: 2)

    Returns:
        str: Formatted number with appropriate suffix
    """
    if number == 0:
        return "0"

    # Handle negative numbers
    is_negative = number < 0
    number = abs(number)

    suffixes = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    small_suffixes = ["", "m", "μ", "n", "p", "f", "a", "z", "y"]

    # For very small numbers (< 1), use different approach
    if number < 1:
        suffix_index = 0

        while number < 1 and suffix_index < len(small_suffixes) - 1:
            number *= 1000
            suffix_index += 1

        formatted = f"{number:.{precision}f} {small_suffixes[suffix_index]}"
    else:
        # For numbers >= 1
        suffix_index = 0

        while number >= 1000 and suffix_index < len(suffixes) - 1:
            number /= 1000
            suffix_index += 1

        formatted = f"{number:.{precision}f} {suffixes[suffix_index]}"

    # Remove trailing zeros and decimal point if not needed
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")

    return f"-{formatted}" if is_negative else formatted


def format_bytes(bytes_value, precision=2):
    """
    Format bytes using binary suffixes (B, KB, MB, GB, etc.).

    Args:
        bytes_value (int, float): The number of bytes
        precision (int): Number of decimal places to show (default: 2)

    Returns:
        str: Formatted bytes with appropriate suffix
    """
    if bytes_value == 0:
        return "0 B"

    try:
        is_negative = bytes_value < 0
    except Exception as e:
        return "Invalid input: " + str(e)
    bytes_value = abs(bytes_value)

    suffixes = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    suffix_index = 0

    while bytes_value >= 1024 and suffix_index < len(suffixes) - 1:
        bytes_value /= 1024
        suffix_index += 1

    formatted = f"{bytes_value:.{precision}f} {suffixes[suffix_index]}"

    # Remove trailing zeros
    if "." in formatted.split()[0]:
        number_part = formatted.split()[0].rstrip("0").rstrip(".")
        formatted = f"{number_part} {suffixes[suffix_index]}"

    return f"-{formatted}" if is_negative else formatted


def format_scientific(number, max_precision=2):
    """
    Format to scientific notation with automatic precision based on number magnitude.

    Args:
        number (int, float): The number to format
        max_precision (int): Maximum number of decimal places (default: 2)

    Returns:
        str: Number formatted in scientific notation
    """
    if number == 0:
        return "0"

    try:
        if math.isnan(number) or math.isinf(number):
            return str(number)
    except Exception as e:
        return "Invalid input: " + str(e)

    # Calculate exponent
    exponent = math.floor(math.log10(abs(number)))
    mantissa = number / (10**exponent)

    # Determine precision based on mantissa
    if abs(mantissa) >= 10:
        mantissa /= 10
        exponent += 1

    # Auto-adjust precision to avoid trailing zeros
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


def format_number_commas(number, precision=2):
    """
    Format a number with commas as thousands separators.

    Args:
        number (int, float): The number to format
        precision (int): Number of decimal places to show (default: 2)

    Returns:
        str: Number formatted with commas
    """
    if isinstance(number, int):
        return f"{number:,}"
    elif isinstance(number, float):
        return f"{number:,.{precision}f}"
    else:
        logger_error.error(f"{RED}Invalid input type: {type(number)}. Must be int or float.{RESET}")
        return "Invalid input: " + str(number)


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
            notebook_path: Path to the ``.ipynb`` file.
            output_path: Optional custom destination for the converted script.
            append_lines: Optional lines to append to the resulting Python file.

        Returns:
            Path to the created ``.py`` file.

        Raises:
            ImportError: If the ``nbformat`` dependency is missing.
            ValueError: If the notebook fails to convert or write to disk.
        """
        try:
            import nbformat
        except ImportError as e:
            raise ImportError("Missing notebook dependencies. " "Please install: pip install nbformat") from e

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
            notebook_path: Path to the ``.ipynb`` file.
            success_flag: Path where the monitored script should write the
                ``SUCCESS`` flag upon completion.

        Returns:
            Path to the generated monitored Python script.

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
