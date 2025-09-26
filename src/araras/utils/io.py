from araras.core import *

import os
import ipynbname
import inspect
import sys
from pathlib import Path


def create_run_directory(prefix: str, base_dir: str = "runs") -> str:
    """Create a run directory with an incremented numeric suffix.

    Args:
        prefix: Prefix used in the directory name (for example ``"run"``).
        base_dir: Parent directory where run folders should be created. The
            folder is created when it does not already exist.

    Returns:
        str: Path to the newly created directory relative to ``base_dir``.

    Examples:
        >>> create_run_directory("run")
        'runs/run1'
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


def get_caller_stem(remove: Optional[str] = "temp_monitor_") -> str:
    """Return the stem name of the script or notebook that invoked this helper.

    Args:
        remove: Optional substring to remove from the detected stem.

    Returns:
        str: Stem name derived from VS Code, Python script or notebook
        metadata.

    Raises:
        RuntimeError: If a stem cannot be inferred from the call stack or
        runtime metadata.
    """

    def clean(stem: str) -> str:
        """Helper to strip out the `remove` substring if requested."""
        return stem.replace(remove, "") if remove else stem

    # inspect the caller’s globals
    caller_globals = inspect.currentframe().f_back.f_globals

    # 1. VS Code notebook: __vsc_ipynb_file__ is set automatically
    vsc_path = caller_globals.get("__vsc_ipynb_file__")
    if vsc_path:
        return clean(Path(vsc_path).stem)

    # 2. Normal .py script: __file__ is defined
    file_path = caller_globals.get("__file__")
    if file_path:
        return clean(Path(file_path).stem)

    # 3. (Optional) pure notebook via ipynbname, if installed
    try:
        ipynbname_stem = Path(ipynbname.path()).stem
        if ipynbname_stem:
            return clean(ipynbname_stem)
    except Exception:
        pass

    # 4. Fallback to sys.argv[0], skipping ipykernel launcher stub
    name = Path(sys.argv[0] or "").stem
    if name and "ipykernel_launcher" not in name:
        return clean(name)

    # If all else fails, raise an error
    raise RuntimeError("Could not determine the caller's stem name.")
