from typing import Iterable, Optional

import os
import ipynbname
import inspect
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path


def create_run_directory(prefix: str, base_dir: str = "runs") -> str:
    """Create a run directory with an incremented numeric suffix.

    Args:
        prefix (str): Prefix used in the directory name (for example ``"run"``).
        base_dir (str): Parent directory where run folders should be created. The
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
        remove (Optional[str]): Optional substring to remove from the detected stem.

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


def select_path(
    select_dir: bool = True,
    extensions: Optional[Iterable[str]] = None,
    description: str = "Select a path",
    initial_dir: Optional[str] = None,
) -> str:
    """Open a native file/folder picker dialog and return the selected path.

    Args:
        select_dir (bool): Controls which dialog mode is used. If ``True``, opens
            a directory chooser and ignores ``extensions`` because folder dialogs
            do not support extension filtering. If ``False``, opens a file chooser
            and applies ``extensions`` as file type filters when provided.
        extensions (Optional[Iterable[str]]): Iterable of extensions to allow when
            ``select_dir`` is ``False``. Extensions may include or omit the leading
            dot (for example ``"csv"`` and ``".csv"`` are both accepted). If this
            is ``None`` or empty, all file types are shown. When ``select_dir`` is
            ``True``, this argument has no effect.
        description (str): Dialog title shown to the user, for example
            ``"Select the dataset root folder dir"``.
        initial_dir (Optional[str]): Optional directory that the dialog opens in.
            If ``None``, the dialog starts at the filesystem root (``/`` on
            Linux/macOS, drive root on Windows) to make disk and mount-point
            selection easier.

    Returns:
        str: The selected absolute path as returned by the dialog. Returns an empty
        string when the user cancels the selection.

    Raises:
        RuntimeError: If the graphical dialog cannot be opened in the current
            environment (for example, missing display server in a headless session).

    Examples:
        >>> dataset_dir = select_path(
        ...     select_dir=True,
        ...     description="Select the dataset root folder dir",
        ... )
        >>> config_file = select_path(
        ...     select_dir=False,
        ...     extensions=["yaml", "yml"],
        ...     description="Select the model config file",
        ... )
    """
    # Normalize extensions once so the file dialog receives predictable wildcard
    # patterns and users can pass either "csv" or ".csv" interchangeably.
    normalized_extensions = [
        f".{ext.lstrip('.').lower()}"
        for ext in (extensions or [])
        if isinstance(ext, str) and ext.strip()
    ]

    # Default to the filesystem root when no initial directory is given so the
    # chooser starts from a location where all mounted disks are reachable.
    dialog_initial_dir = initial_dir if initial_dir is not None else os.path.abspath(os.sep)

    # On Linux, prefer the desktop-native chooser when available because it
    # exposes device entries (for example, multiple SSDs) in the side panel.
    # We intentionally keep tkinter as a fallback to preserve cross-platform use.
    if sys.platform.startswith("linux") and shutil.which("zenity"):
        zenity_command = [
            "zenity",
            "--file-selection",
            f"--title={description}",
            f"--filename={dialog_initial_dir.rstrip(os.sep) + os.sep}",
        ]

        if select_dir:
            zenity_command.append("--directory")
        elif normalized_extensions:
            wildcard_group = " ".join(f"*{ext}" for ext in normalized_extensions)
            zenity_command.append(f"--file-filter=Allowed files | {wildcard_group}")

        try:
            result = subprocess.run(
                zenity_command,
                check=False,
                capture_output=True,
                text=True,
            )

            # Exit status 0 means a path was selected; status 1 means user canceled.
            if result.returncode == 0:
                return result.stdout.strip()
            if result.returncode == 1:
                return ""
            # Any other status indicates an execution issue; fallback below.
        except Exception:
            # If zenity fails unexpectedly, fallback to tkinter to avoid breaking
            # callers in environments with partial desktop integration.
            pass

    try:
        # Create a minimal hidden root window because tkinter dialogs require an
        # active root, then hide it to avoid displaying an extra empty window.
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if select_dir:
            # Directory mode intentionally ignores extension filters because the
            # platform directory chooser APIs do not apply them.
            selected_path = filedialog.askdirectory(
                title=description,
                initialdir=dialog_initial_dir,
                mustexist=True,
            )
        else:
            # In file mode, apply explicit extension filters when provided and keep
            # an "All files" fallback so users can still navigate freely.
            filetypes = [("All files", "*")]
            if normalized_extensions:
                wildcard_group = " ".join(f"*{ext}" for ext in normalized_extensions)
                filetypes.insert(0, ("Allowed files", wildcard_group))

            selected_path = filedialog.askopenfilename(
                title=description,
                initialdir=dialog_initial_dir,
                filetypes=filetypes,
            )

        root.destroy()
        return str(selected_path or "")
    except Exception as exc:
        raise RuntimeError("Could not open a graphical path selection dialog.") from exc
