"""
This module provides a restarting monitoring system for processes with email alert capabilities.
It monitors a process for crashes and restarts it if necessary.

Usage example:
    run_auto_restart(
        file_path="my_script.py",
        title="My Critical Process",
    )
"""
from araras.commons import *

import os
import sys
import time
import json
import glob
import psutil
import tempfile
import subprocess
from pathlib import Path

from threading import Event, Thread

# Local imports
from araras.utils.cleanup import ChildProcessCleanup
from araras.utils.terminal import SimpleTerminalLauncher
from araras.utils.misc import NotebookConverter, clear


# Enhanced HTML template for consolidated status reports
CONSOLIDATED_STATUS_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:{color}">{status_title}</h2><div style="background:#f9f9f9;padding:15px;margin:15px 0;border-left:4px solid {color}"><h3>Process Information</h3><p><strong>Process:</strong> {title}</p><p><strong>Status:</strong> {status_description}</p><p><strong>Timestamp:</strong> {timestamp}</p></div>{details_section}<div style="background:#f0f0f0;padding:10px;margin-top:20px;font-size:12px;color:#666"><p>This is an automated status report from the process monitoring system.</p></div></div></body></html>"""

RESTART_DETAILS_TEMPLATE = """<div style="background:#fff3cd;padding:15px;margin:15px 0;border-left:4px solid #ffc107"><h3>Restart Information</h3><p><strong>Previous PID:</strong> {old_pid}</p><p><strong>New PID:</strong> {new_pid}</p><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Runtime Before Restart:</strong> {runtime:.1f}s</p></div>"""

FAILURE_DETAILS_TEMPLATE = """<div style="background:#f8d7da;padding:15px;margin:15px 0;border-left:4px solid #dc3545"><h3>Failure Details</h3><p><strong>Failed Attempts:</strong> {failed_attempts}</p><p><strong>Remaining Attempts:</strong> {remaining_attempts}</p><p><strong>Total Restart Count:</strong> {restart_count}</p><p><strong>Error:</strong> {error}</p></div>"""

COMPLETION_DETAILS_TEMPLATE = """<div style="background:#d4edda;padding:15px;margin:15px 0;border-left:4px solid #28a745"><h3>Completion Summary</h3><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Total Runtime:</strong> {total_runtime:.1f}s</p><p><strong>Final Status:</strong> Successfully completed</p></div>"""

# Updated monitoring script with consolidated email capabilities
MONITOR_SCRIPT = """import os,sys,time,psutil,json
sys.path.insert(0,r"{cwd}")

with open(r"{pid_file}", "w") as f:
    f.write(str(os.getpid()))

def send_crash_signal(pid, title, restart_count=0):
    \"\"\"Send crash signal for restart manager to handle.\"\"\"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())
    print(f"CRASH DETECTED: {{title}} (PID {{pid}}) at {{timestamp}}")
    
    with open(r"{restart_file}", "w") as f:
        json.dump({{"crashed": True, "timestamp": timestamp, "restart_count": restart_count, "pid": pid}}, f)
    
    try: os.unlink(r"{pid_file}")
    except: pass
    sys.exit(0)

try:
    proc = psutil.Process({pid})
    print(f"Monitoring PID {{pid}} for crashes")
except psutil.NoSuchProcess:
    send_crash_signal({pid}, {title})

count = 0
while True:
    # Check stop signal every 10 iterations to reduce I/O overhead
    if count % 10 == 0 and os.path.exists(r"{stop_file}"):
        try: os.unlink(r"{pid_file}")
        except: pass
        break
    
    count += 1
    
    try:
        if not proc.is_running():
            restart_count = 0
            try:
                if os.path.exists(r"{restart_file}"):
                    with open(r"{restart_file}") as f:
                        data = json.load(f)
                        restart_count = data.get("restart_count", 0)
            except:
                pass
            send_crash_signal({pid}, {title}, restart_count)
        
        # Check for zombie/stopped states that indicate crashes
        status = proc.status()
        if status in [psutil.STATUS_ZOMBIE, psutil.STATUS_STOPPED, psutil.STATUS_DEAD]:
            restart_count = 0
            try:
                if os.path.exists(r"{restart_file}"):
                    with open(r"{restart_file}") as f:
                        data = json.load(f)
                        restart_count = data.get("restart_count", 0)
            except:
                pass
            send_crash_signal({pid}, {title}, restart_count)

    except psutil.NoSuchProcess:
        restart_count = 0
        try:
            if os.path.exists(r"{restart_file}"):
                with open(r"{restart_file}") as f:
                    data = json.load(f)
                    restart_count = data.get("restart_count", 0)
        except:
            pass
        send_crash_signal({pid}, {title}, restart_count)
    except Exception:
        restart_count = 0
        try:
            if os.path.exists(r"{restart_file}"):
                with open(r"{restart_file}") as f:
                    data = json.load(f)
                    restart_count = data.get("restart_count", 0)
        except:
            pass
        send_crash_signal({pid}, {title}, restart_count)
    
    time.sleep({interval})

print("Monitor completed")"""


# ——————————————————————————— Print Functions ——————————————————————————————— #
ONCE_PRINT = False # Flag to ensure print statements only run once
def print_monitoring_config_summary( 
    file_path: str,
    file_type: str,
    success_flag_file: str,
    max_restarts: int,
    email_enabled: bool,
    title: str,
    restart_after_delay: Optional[float] = None,
) -> None:
    """Print a summary of monitoring configuration only once."""
    global ONCE_PRINT
    if ONCE_PRINT:
        return
    ONCE_PRINT = True

    print()
    print("=" * 70)
    print("MONITORING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Target File: {file_path}")
    print(f"Success Flag Location: {success_flag_file}")
    # print(f"File Type: {file_type}")
    print(f"Process Title: \033[33m{title}\033[0m")
    if email_enabled:
        print(f"Email Alerts: \033[92mEnabled\033[0m")
    else:
        print(f"Email Alerts: \033[91mDisabled\033[0m")
    print(f"Max Restarts: {max_restarts}")
    if restart_after_delay is not None:
        print(f"Run will force restart after: {restart_after_delay} seconds")
    print("=" * 70)
    print()


def print_process_status(message: str, pid: Optional[int] = None, runtime: Optional[float] = None) -> None:
    """Print process status messages with consistent formatting."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    if pid and runtime is not None:
        print(f"[{timestamp}] {message} (PID {pid}, runtime: {runtime:.1f}s)")
    elif pid:
        print(f"[{timestamp}] {message} \033[33m(PID {pid})\033[0m")
    else:
        print(f"[{timestamp}] {message}")


def print_restart_info(restart_count: int, max_restarts: int, delay: float) -> None:
    """Print restart information with formatting."""
    print_process_status(f"Restarting in {delay:.1f}s \033[33m({restart_count}/{max_restarts})\033[0m")


def print_completion_summary(restart_count: int, total_runtime: Optional[float] = None) -> None:
    """Print final completion summary."""
    # print("\n" + "=" * 50)
    print("\n\033[Process Completed\033[0m")
    # print("=" * 50)
    print(f"Total Restarts: \033[33m{restart_count}\033[0m")
    # if total_runtime is not None:
    #     print(f"Total Runtime:  {total_runtime:.1f}s")
    # print("=" * 50)


def print_error_message(error_type: str, message: str) -> None:
    """Print error messages with consistent formatting."""
    print(f"ERROR [{error_type}]: {message}")


def print_warning_message(message: str) -> None:
    """Print warning messages with consistent formatting."""
    print(f"Warning: {message}")


def print_success_message(message: str) -> None:
    """Print success messages with consistent formatting."""
    print(f"SUCCESS: {message}")


def print_cleanup_info(terminated: int, killed: int) -> None:
    """Print child process cleanup information."""
    if terminated > 0 or killed > 0:
        print(f"Child cleanup: {terminated} terminated, {killed} killed")


# —————————————————————————————————— Utility ————————————————————————————————— #
def _cleanup_stale_monitor_files():
    tmpdir = tempfile.gettempdir()
    for path in glob.glob(os.path.join(tmpdir, "*_monitor.*")):
        try:
            os.unlink(path)
        except OSError:
            pass


def get_process_resource_usage(pid: int) -> Tuple[float, float, float]:
    """Return memory percentage, memory in GB, and CPU percentage for a process.

    Args:
        pid: Process ID of the process to query.

    Returns:
        Tuple containing memory percentage, memory usage in GB and CPU percentage.

    Raises:
        psutil.NoSuchProcess: If the PID does not exist.
    """
    proc = psutil.Process(pid)

    # First call to initialize CPU measurement (returns 0.0)
    proc.cpu_percent()

    # Wait and take actual measurement
    time.sleep(1.0)  # Longer interval for accurate measurement

    with proc.oneshot():
        mem_percent = proc.memory_percent()
        mem_gb = proc.memory_info().rss / (1024**3)
        cpu_percent = proc.cpu_percent()  # Non-blocking call after initialization
        
    # write to a log file in the root directory
    log_file = os.path.join(os.getcwd(), "process_resource_usage.log")
    with open(log_file, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        f.write(f"Time: {timestamp}, PID: {pid}, MEM: {mem_percent:.2f}%, {mem_gb:.2f} GB, CPU: {cpu_percent:.2f}%\n")

    return mem_percent, mem_gb, cpu_percent


def print_process_resource_usage(pid: int) -> None:
    """Display CPU and memory usage for a process in a single updating line."""
    try:
        mem_p, mem_gb, cpu_p = get_process_resource_usage(pid)
        print(
            f"CPU:{cpu_p:5.1f}% MEM:{mem_p:5.1f}% ({mem_gb:.2f} GB)".ljust(60),
            end="\r",
            flush=True,
        )
    except Exception:
        pass

from .consolidated_email_manager import ConsolidatedEmailManager
from .file_type_handler import FileTypeHandler
from .flag_based_restart_manager import FlagBasedRestartManager


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
                            # clear()

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
                    print("\n")
                    print_process_status(
                        "Restart loop interrupted by user, cleaning up. \033[91mPlease wait...\033[0m"
                    )
                    manager.force_stop()
                    manager._cleanup_converted_file()
                # Ensure the worker thread has completely finished before returning
                thread.join()
                print(f"\n\033[92mProcess done!\033[0m")

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
