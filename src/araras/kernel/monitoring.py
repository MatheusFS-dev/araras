"""
This module provides a restarting monitoring system for processes with email alert capabilities.
It monitors a process for crashes and restarts it if necessary.

Usage example:
    run_auto_restart(
        file_path="my_script.py",
        title="My Critical Process",
    )
"""

import os
import sys
import time
import json
import glob
import psutil
import tempfile
import subprocess
from typing import *
from pathlib import Path

from threading import Event, Thread

# Local imports
from araras.utils.cleanup import ChildProcessCleanup
from araras.utils.terminal import SimpleTerminalLauncher
from araras.utils.misc import NotebookConverter, clear

from .print_utils import (
    print_monitoring_config_summary,
    print_process_status,
    print_restart_info,
    print_completion_summary,
    print_error_message,
    print_warning_message,
    print_success_message,
    print_cleanup_info,
)
from .email_manager import ConsolidatedEmailManager
from .file_type_handler import FileTypeHandler
from .restart_manager import FlagBasedRestartManager
from .templates import (
    CONSOLIDATED_STATUS_TEMPLATE,
    RESTART_DETAILS_TEMPLATE,
    FAILURE_DETAILS_TEMPLATE,
    COMPLETION_DETAILS_TEMPLATE,
    MONITOR_SCRIPT,
)


    


def start_monitor(pid: int, title: str, supress_tf_warnings: bool = False) -> Dict[str, Any]:
    """Start simplified crash monitor without email capabilities.

    Args:
        pid: Process ID to monitor
        title: Process title for alerts
        supress_tf_warnings: Suppress TensorFlow warnings (default: False)

    Returns:
        Monitor control info dictionary

    Raises:
        ValueError: If PID doesn't exist
        OSError: If monitor startup fails
    """
    _cleanup_stale_monitor_files()
    time.sleep(0.1)  # Allow time for process to stabilize

    if not psutil.pid_exists(pid):
        raise ValueError(f"Process PID {pid} not found")

    # Create minimal control files
    fd, script_path = tempfile.mkstemp(suffix="_monitor.py")
    base_path = script_path.replace(".py", "")

    control_files = {
        "script_path": script_path,
        "pid_file": f"{base_path}.pid",
        "stop_file": f"{base_path}.stop",
        "restart_file": f"{base_path}.restart",
    }

    # Generate simplified monitoring script
    script_content = MONITOR_SCRIPT.format(
        cwd=os.getcwd(),
        pid=pid,
        interval=2,
        title=repr(title),
        **control_files,
    )

    with os.fdopen(fd, "w") as f:
        f.write(script_content)

    if os.name != "nt":
        os.chmod(script_path, 0o755)

    # Launch monitor in terminal
    launcher = SimpleTerminalLauncher()
    launcher.set_supress_tf_warnings(supress_tf_warnings)
    process = launcher.launch([sys.executable, script_path], os.getcwd())

    time.sleep(0.1)
    if process.poll() is not None:  # Check if it died
        exit_code = process.returncode
        error_msg = f"Monitor failed to start (exit code: {exit_code})"

        # Try to get stderr output if available
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stderr:
                error_msg += f". Error output: {stderr.decode().strip()}"
            elif stdout:
                error_msg += f". Output: {stdout.decode().strip()}"
        except:
            pass

        # Cleanup the failed script file
        try:
            os.unlink(script_path)
        except:
            pass

        raise OSError(error_msg)

    return {"process": process, **control_files}


def stop_monitor(monitor_info: Dict[str, Any]) -> None:
    """Stop monitor and cleanup files with optimized batch operations.

    Args:
        monitor_info: Monitor control info from start_monitor()
    """
    if not monitor_info:
        return

    # Signal stop (single I/O operation)
    try:
        with open(monitor_info["stop_file"], "w") as f:
            f.write("STOP")
    except:
        pass

    # Wait for graceful shutdown with optimized timeout
    for _ in range(20):  # 2 second timeout
        if not os.path.exists(monitor_info["pid_file"]):
            break
        time.sleep(0.1)

    # Force terminate if needed
    process = monitor_info.get("process")
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            pass

    # Batch file cleanup (single loop for efficiency)
    cleanup_files = ["script_path", "pid_file", "stop_file", "restart_file"]
    for file_key in cleanup_files:
        try:
            file_path = monitor_info.get(file_key)
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass


def check_crash_signal(monitor_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check if process crashed with minimal I/O operations.

    Args:
        monitor_info: Monitor control info

    Returns:
        Dictionary with crash info or empty dict if no crash
    """
    restart_file = monitor_info.get("restart_file")
    if not restart_file or not os.path.exists(restart_file):
        return {}

    try:
        with open(restart_file) as f:
            data = json.load(f)
            if data.get("crashed", False):
                return data
    except:
        pass

    return {}


def run_auto_restart(
    file_path: str,
    success_flag_file: str = "/tmp/success.flag",
    title: Optional[str] = None,
    max_restarts: int = 10,
    restart_delay: float = 3.0,
    recipients_file: Optional[str] = None,
    credentials_file: Optional[str] = None,
    restart_after_delay: Optional[float] = None,
    retry_attempts: int = None,
    supress_tf_warnings: bool = False,
) -> None:
    """Main function with notebook conversion, file cleanup, and consolidated email notification support.

    Args:
        file_path: Path to .py or .ipynb file to execute
        success_flag_file: Path to success flag file
        title: Custom title for monitoring and email alerts
        max_restarts: Maximum restart attempts
        restart_delay: Delay between restarts in seconds
        recipients_file: Path to recipients JSON file (defaults to ./json/recipients.json)
        credentials_file: Path to credentials JSON file (defaults to ./json/credentials.json)
        restart_after_delay: restart the run after a delay in seconds
        retry_attempts: Number of retry attempts before sending failure email
        supress_tf_warnings: Suppress TensorFlow warnings (default: False)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported
        ImportError: If notebook dependencies missing for .ipynb files
    """

    try:
        # Clean up any existing success flag file before starting
        Path(success_flag_file).unlink(missing_ok=True)

        manager = FlagBasedRestartManager(
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
            retry_attempts=max_restarts if retry_attempts is None else retry_attempts,
        )

        if restart_after_delay is not None and restart_after_delay > 0:
            # Wrapping logic for forced restart not counting as crash/max_restarts
            # This will run in a loop, restarting after each interval, until success_flag is found.

            stop_event = Event()

            def restart_loop():
                try:
                    while not stop_event.is_set():
                        manager.restart_count = 0  # Never increment max_restarts for forced restart
                        # Ensure no leftover processes remain running
                        manager._cleanup_stale_pids()
                        finished = [False]

                        def run_and_flag():
                            try:
                                manager.run_file_with_restart(
                                    file_path=file_path,
                                    success_flag_file=success_flag_file,
                                    title=title,
                                    restart_after_delay=restart_after_delay,
                                    supress_tf_warnings=supress_tf_warnings,
                                )
                                finished[0] = True
                            except Exception:
                                finished[0] = True  # On error, still allow restart

                        thread = Thread(target=run_and_flag)
                        thread.start()
                        thread.join(timeout=restart_after_delay)
                        if thread.is_alive():
                            print_process_status(
                                f"Forcing restart after {restart_after_delay} seconds (not a crash)"
                            )
                            manager.force_stop()
                            # Ensure the worker thread finishes cleanly before continuing
                            thread.join(5)
                            clear()

                        else:
                            # If finished (success or crash), check if success
                            if Path(success_flag_file).exists():
                                stop_event.set()
                            else:
                                print_process_status(
                                    "Process ended before restart_after_delay, restarting..."
                                )
                except KeyboardInterrupt:
                    # Handle CTRL+C in the restart loop
                    stop_event.set()
                    print_process_status("Restart loop interrupted by user, cleaning up")
                    manager.force_stop()
                    manager._cleanup_converted_file()
                # Ensure the worker thread has completely finished before returning
                thread.join()
                print_process_status("Restart-after-delay loop done")

            restart_loop()

        else:
            # Regular auto-restart logic
            manager.run_file_with_restart(
                file_path=file_path,
                success_flag_file=success_flag_file,
                title=title,
                supress_tf_warnings=supress_tf_warnings,
            )

    except (FileNotFoundError, ValueError, ImportError) as e:
        print_error_message("CONFIG", str(e))
        raise
    except KeyboardInterrupt:
        print_process_status("Main process interrupted by user, performing final cleanup")
    except Exception as e:
        print_error_message("FATAL", str(e))
        raise
