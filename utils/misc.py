"""
Miscellaneous utility functions for the Araras project.

Functions:
    - clear_terminal: Clears the terminal or notebook cell output, compatible with various environments.
    - format_number: Formats a number with appropriate suffixes (K, M, G, etc.) and precision.
    - format_bytes: Formats a byte value with binary suffixes (B, KB, MB, etc.) and precision.
    - format_scientific: Formats a number in scientific notation with automatic precision.
    - format_number_commas: Formats a number with commas as thousands separators.
"""

import os
import math
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
        print(f"Error clearing terminal: {e}")


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

    is_negative = bytes_value < 0
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

    if math.isnan(number) or math.isinf(number):
        return str(number)

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
        raise ValueError("Input must be an integer or float")