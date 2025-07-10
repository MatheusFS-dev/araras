"""Utility print functions used by the monitoring system."""

from __future__ import annotations

import time
from typing import Optional

__all__ = [
    "print_monitoring_config_summary",
    "print_process_status",
    "print_restart_info",
    "print_completion_summary",
    "print_error_message",
    "print_warning_message",
    "print_success_message",
    "print_cleanup_info",
]


def print_monitoring_config_summary(
    file_path: str,
    success_flag_file: str,
    max_restarts: int,
    email_enabled: bool,
    title: str,
    restart_after_delay: Optional[float] = None,
) -> None:
    """Print a summary of monitoring configuration without file type."""
    print()
    print("=" * 70)
    print("MONITORING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Target File: {file_path}")
    print(f"Process Title: \033[33m{title}\033[0m")
    print(f"Success Flag: {success_flag_file}")
    print(f"Max Restarts: {max_restarts}")
    if restart_after_delay is not None:
        print(f"Run will force restart after: {restart_after_delay} seconds")

    if email_enabled:
        print(f"Email Alerts: \033[92mEnabled\033[0m")
    else:
        print(f"Email Alerts: \033[91mDisabled\033[0m")
    print("=" * 70)
    print()


def print_process_status(message: str, pid: Optional[int] = None, runtime: Optional[float] = None) -> None:
    """Print process status messages with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    if pid and runtime is not None:
        print(f"[{timestamp}] {message} (PID {pid}, runtime: {runtime:.1f}s)")
    elif pid:
        print(f"[{timestamp}] {message} (PID {pid})")
    else:
        print(f"[{timestamp}] {message}")


def print_restart_info(restart_count: int, max_restarts: int, delay: float) -> None:
    """Print restart information."""
    print(f"Restarting in {delay:.1f}s ({restart_count}/{max_restarts})")


def print_completion_summary(restart_count: int, total_runtime: Optional[float] = None) -> None:
    """Print final completion summary."""
    print("=" * 50)
    print("MONITORING COMPLETED")
    print("=" * 50)
    print(f"Total Restarts: {restart_count}")
    if total_runtime is not None:
        print(f"Total Runtime:  {total_runtime:.1f}s")
    print("=" * 50)


def print_error_message(error_type: str, message: str) -> None:
    """Print an error message."""
    print(f"ERROR [{error_type}]: {message}")


def print_warning_message(message: str) -> None:
    """Print a warning message."""
    print(f"Warning: {message}")


def print_success_message(message: str) -> None:
    """Print a success message."""
    print(f"SUCCESS: {message}")


def print_cleanup_info(terminated: int, killed: int) -> None:
    """Print child process cleanup information."""
    if terminated > 0 or killed > 0:
        print(f"Child cleanup: {terminated} terminated, {killed} killed")
