"""
This module provides utility functions for managing directories, such as creating
incrementally named run directories for experiments or other purposes.

Functions:
    - create_run_directory: Creates a new directory with an incremented numeric suffix.

Example usage:
    run_dir = create_run_directory(prefix="run")
    print(run_dir)  # outputs: runs/run1, runs/run2, etc.
"""

import os


def create_run_directory(prefix: str, base_dir: str = "runs") -> str:
    """
    Creates a new run directory with an incremented numeric suffix and returns its full path.

    The directory name is generated using the given prefix followed by the next available number.
    For example, if directories "run1", "run2", and "run3" exist, calling with prefix="run" will create "run4".

    Logic:
        -> Ensure base_dir exists
        -> List existing directories with matching prefix and numeric suffix
        -> Parse suffix numbers and find the next available integer
        -> Construct full path using prefix + next number
        -> Create the new run directory and return its path

    Args:
        prefix (str): Prefix to be used in the name of each run directory (e.g., "run").
        base_dir (str, optional): Directory under which all runs are stored. Defaults to "runs".

    Returns:
        str: Absolute path to the newly created run directory.

    Example:
        run_path = create_run_directory(prefix="run")
        print(run_path)  # outputs: runs/run1, runs/run2, etc.
    """
    # Ensure the base directory exists; create it if it doesn't
    os.makedirs(base_dir, exist_ok=True)

    # Collect numeric suffixes of existing directories that match the given prefix
    existing = [
        int(d[len(prefix) :])  # Extract numeric part after prefix
        for d in os.listdir(base_dir)  # Iterate over all entries in base_dir
        if d.startswith(prefix) and d[len(prefix) :].isdigit()  # Ensure suffix is all digits
    ]

    # Determine the next run number: max existing number + 1, or 1 if none exist
    run_number = (max(existing) if existing else 0) + 1

    # Build the new directory path using prefix and computed run number
    run_dir = os.path.join(base_dir, f"{prefix}{run_number}")

    # Create the new run directory
    os.makedirs(run_dir)

    # Return the full path to the created directory
    return run_dir
