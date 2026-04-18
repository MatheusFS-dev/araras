"""
This script auto launches the main application and restarts it if it crashes.

Command line usage:
    python _auto_run.py nas_cnn1d_flat_v9.ipynb --title "ML Training"

    For Anaconda environments, use:
    /home/matheus/anaconda3/envs/tf-optuna/bin/python _auto_run.py nas_cnn1d_flat_v9.ipynb --title "ML Training"
"""
from typing import List, Optional

import sys
import argparse
from araras.runtime.monitoring import DEFAULT_SUCCESS_FLAG_FILE, run_auto_restart

from araras.utils.verbose_printer import VerbosePrinter

vp = VerbosePrinter()

# ———————————————————— Warnings and Executable Information ——————————————————— #
vp.printf(vp.color("=" * 80, "yellow"))
vp.printf(vp.color("MONITOR SCRIPT".center(80), "yellow"))
vp.printf(vp.color("=" * 80, "yellow"))
vp.printf(vp.color("Run this script with ", "yellow") + vp.color("sudo", "red") + vp.color(" if the other process requires it!", "yellow"))
vp.printf(vp.color("If using a Conda Venv, ensure you are using the correct python interpreter.", "yellow"))
vp.printf(vp.color(f">>> {sys.executable} is running this script", "orange"))
vp.printf(vp.color("=" * 80, "yellow"))
# ———————————————————————————————————————————————————————————————————————————— #


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the monitoring CLI.

    This helper assembles the ``argparse`` parser used by the CLI
    entry point. Most options provide both short and long forms for
    convenience. It also accepts multiple target files which are
    monitored sequentially.

    Args:
        argv (Optional[List[str]]): Optional list of arguments to parse instead of ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run a Python script or notebook with automatic restarts",
    )

    parser.add_argument(
        "file_paths",
        nargs="+",
        help="Path(s) to the .py or .ipynb files to execute sequentially",
    )
    parser.add_argument(
        "-s",
        "--success-flag-file",
        default=DEFAULT_SUCCESS_FLAG_FILE,
        help=(
            "Path where the executed script writes a completion flag. "
            "If omitted a unique file in the system temporary directory is used."
        ),
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        help="Custom title for monitoring and email alerts",
    )
    parser.add_argument(
        "-m",
        "--max-restarts",
        type=int,
        default=10,
        help="Maximum number of restart attempts",
    )
    parser.add_argument(
        "-d",
        "--restart-delay",
        type=float,
        default=3.0,
        help="Delay between restarts in seconds",
    )
    parser.add_argument(
        "-r",
        "--recipients-file",
        default=None,
        help="Path to JSON file containing email recipients",
    )
    parser.add_argument(
        "-c",
        "--credentials-file",
        default=None,
        help="Path to JSON file with email credentials",
    )
    parser.add_argument(
        "-f",
        "--force-restart",
        type=float,
        default=None,
        help="Force a restart after this many seconds regardless of status",
    )
    parser.add_argument(
        "-a",
        "--retry-attempts",
        type=int,
        default=None,
        help="Number of retry attempts before a failure email is sent",
    )
    parser.add_argument(
        "-w",
        "--supress-tf-warnings",
        action="store_true",
        default=False,
        help="Suppress TensorFlow warnings",
    )
    parser.add_argument(
        "--no-restart-email",
        action="store_true",
        default=False,
        help="Disable email warnings for process restarts",
    )
    parser.add_argument(
        "-u",
        "--resource-usage-log-file",
        default=None,
        help="File to log process resource usage statistics",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``monitor`` command.

    This wrapper parses command line flags and forwards them to
    :func:`run_auto_restart`. It is exposed as ``monitor`` via the
    package's ``pyproject.toml`` entry point and can also be invoked with
    ``python -m araras.runtime.monitoring``. When multiple ``file_paths``
    are provided they are executed sequentially.

    Note:
        Execute this command from the **same directory** as the script or
        notebook being monitored so that relative paths resolve correctly.

    Args:
        argv (Optional[List[str]]): Optional list of arguments to parse instead of ``sys.argv``.

    Returns:
        None: ``None``

    Raises:
        SystemExit: If invalid options are supplied and argument parsing
        fails.
    """

    args = _parse_args(argv)
    for target in args.file_paths:
        run_auto_restart(
            file_path=target,
            success_flag_file=args.success_flag_file,
            title=args.title,
            max_restarts=args.max_restarts,
            restart_delay=args.restart_delay,
            recipients_file=args.recipients_file,
            credentials_file=args.credentials_file,
            force_restart=args.force_restart,
            retry_attempts=args.retry_attempts,
            supress_tf_warnings=args.supress_tf_warnings,
            restart_email_warning=not args.no_restart_email,
            resource_usage_log_file=args.resource_usage_log_file,
        )


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
