from typing import Any, Dict, Optional, Tuple

import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil

# Local imports
from .file_handler import FileTypeHandler

from araras.utils.verbose_printer import VerbosePrinter

DEFAULT_SUCCESS_FLAG_FILE = "/tmp/success.flag"

vp = VerbosePrinter()

# Path where resource usage logs will be written. If ``None`` no logging occurs
RESOURCE_USAGE_LOG_FILE: Optional[str] = None

# Enhanced HTML template for consolidated status reports
CONSOLIDATED_STATUS_TEMPLATE = """<html><body style="font-family:Arial,sans-serif;color:#333;padding:20px"><div style="max-width:600px;margin:auto;background:#fff;padding:20px;border:1px solid #ddd"><h2 style="color:{color}">{status_title}</h2><div style="background:#f9f9f9;padding:15px;margin:15px 0;border-left:4px solid {color}"><h3>Process Information</h3><p><strong>Process:</strong> {title}</p><p><strong>Status:</strong> {status_description}</p><p><strong>Timestamp:</strong> {timestamp}</p></div>{details_section}<div style="background:#f0f0f0;padding:10px;margin-top:20px;font-size:12px;color:#666"><p>This is an automated status report from the process monitoring system.</p></div></div></body></html>"""

RESTART_DETAILS_TEMPLATE = """<div style="background:#fff3cd;padding:15px;margin:15px 0;border-left:4px solid #ffc107"><h3>Restart Information</h3><p><strong>Previous PID:</strong> {old_pid}</p><p><strong>New PID:</strong> {new_pid}</p><p><strong>Total Restarts:</strong> {restart_count}</p><p><strong>Runtime Before Restart:</strong> {runtime:.1f}s</p></div>"""

SCHEDULED_RESTART_DETAILS_TEMPLATE = """<div style="background:#d4edda;padding:15px;margin:15px 0;border-left:4px solid #28a745"><h3>Scheduled Restart Information</h3><p><strong>Previous PID:</strong> {old_pid}</p><p><strong>New PID:</strong> {new_pid}</p><p><strong>Scheduled Restart Count:</strong> {scheduled_restart_count}</p><p><strong>Configured Interval:</strong> {interval_minutes:g} minutes</p><p><strong>Runtime Before Restart:</strong> {runtime:.1f}s</p><img src="cid:monitor-graph" alt="Previous process-tree RAM over time" style="max-width:100%;height:auto" /></div>"""

MEMORY_AWARE_SCHEDULED_RESTART_DETAILS_TEMPLATE = """<div style="background:#d4edda;padding:15px;margin:15px 0;border-left:4px solid #28a745"><h3>Memory-Aware Restart Information</h3><p><strong>Previous PID:</strong> {old_pid}</p><p><strong>New PID:</strong> {new_pid}</p><p><strong>Scheduled Restart Count:</strong> {scheduled_restart_count}</p><p><strong>Triggering Process-Tree RSS:</strong> {current_rss_gib:.2f} GiB</p><p><strong>Configured RSS Threshold:</strong> {threshold_gib:.2f} GiB</p><p><strong>Polling Interval:</strong> {poll_minutes:g} minutes</p><p><strong>Runtime Before Restart:</strong> {runtime:.1f}s</p>{graph_section}</div>"""

FAILURE_DETAILS_TEMPLATE = """<div style="background:#f8d7da;padding:15px;margin:15px 0;border-left:4px solid #dc3545"><h3>Failure Details</h3><p><strong>Failed Attempts:</strong> {failed_attempts}</p><p><strong>Remaining Attempts:</strong> {remaining_attempts}</p><p><strong>Total Restart Count:</strong> {restart_count}</p><p><strong>Error:</strong> {error}</p></div>"""

COMPLETION_DETAILS_TEMPLATE = """<div style="background:#d4edda;padding:15px;margin:15px 0;border-left:4px solid #28a745"><h3>Completion Summary</h3><p><strong>Total Restarts:</strong> {total_restarts}</p><p><strong>Crash/Failure Restarts:</strong> {restart_count}</p><p><strong>Scheduled Restarts:</strong> {scheduled_restart_count}</p><p><strong>Total Runtime:</strong> {total_runtime:.1f}s</p><p><strong>Final Status:</strong> Successfully completed</p></div>"""

MEMORY_LEAK_DETAILS_TEMPLATE = """<div style="background:#fff3cd;padding:15px;margin:15px 0;border-left:4px solid #b54708"><h3>Memory Trend Evidence</h3><p><strong>PID:</strong> {pid}</p><p><strong>Current Process-Tree RSS:</strong> {current_rss_mib:.1f} MiB</p><p><strong>Net Growth:</strong> {net_growth_mib:.1f} MiB</p><p><strong>Growth Slope:</strong> {slope_mib_per_minute:.1f} MiB/min</p><p><strong>R-squared (diagnostic only):</strong> {r_squared:.3f}</p><p><strong>Warm-up:</strong> {warmup_minutes:g} minutes</p><p><strong>Analysis Window:</strong> {window_minutes:g} minutes</p><p>This trend may indicate a memory leak, but RSS growth alone cannot prove one.</p><img src="cid:monitor-graph" alt="Process-tree RAM over time" style="max-width:100%;height:auto" /></div>"""

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
    print("Monitoring PID {pid} for crashes")
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
ONCE_PRINT = False  # Flag to ensure print statements only run once


def print_monitoring_config_summary(
    file_path: str,
    file_type: str,
    success_flag_file: str,
    max_restarts: int,
    email_enabled: bool,
    title: str,
    restart_after_delay: Optional[float] = None,
    scheduled_restart_memory_threshold_bytes: Optional[float] = None,
    scheduled_restart_poll_interval_seconds: Optional[float] = None,
    detect_memory_leaks: bool = False,
    memory_leak_warmup_seconds: float = 300.0,
) -> None:
    """Print a summary of monitoring configuration only once.

    This function outputs a one-time overview of the monitoring setup for a
    target file. Subsequent calls are ignored to avoid repeated messages.

    Args:
        file_path (str): Path of the file displayed in the summary.
        file_type (str): Detected type of the monitored file.
        success_flag_file (str): Location where the SUCCESS flag is expected.
        max_restarts (int): Maximum restart attempts allowed.
        email_enabled (bool): Whether email notifications are enabled.
        title (str): Title shown for the monitored process.
        restart_after_delay (Optional[float]): Optional forced restart delay in seconds.
        scheduled_restart_memory_threshold_bytes (Optional[float]): Optional
            target-process-tree RSS threshold in bytes. When provided with
            ``scheduled_restart_poll_interval_seconds``, scheduled restarts
            are deferred until the threshold is reached.
        scheduled_restart_poll_interval_seconds (Optional[float]): Optional
            delay in seconds between deferred smart-restart RSS checks. It
            must be provided with
            ``scheduled_restart_memory_threshold_bytes``.
        detect_memory_leaks (bool): Whether conservative process-tree RSS trend
            detection is enabled. Defaults to ``False``.
        memory_leak_warmup_seconds (float): Initial no-check duration in
            seconds. Displayed only when detection is enabled. Defaults to
            ``300.0``.

    Notes:
        This function only prints configuration information and does not alter
        monitoring behavior.

    Raises:
        ValueError: If only one smart scheduled-restart setting is supplied.
    """
    if (
        scheduled_restart_memory_threshold_bytes is None
    ) != (
        scheduled_restart_poll_interval_seconds is None
    ):
        raise ValueError(
            "scheduled restart memory threshold and polling interval must be provided together"
        )

    global ONCE_PRINT
    if ONCE_PRINT:
        return
    ONCE_PRINT = True

    print()
    print("=" * 80)
    print("MONITORING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Target File: {file_path}")
    print(f"Success Flag Location: {success_flag_file}")
    # print(f"File Type: {file_type}")
    print("Process Title: " + vp.color(f"{title}", "orange"))
    if email_enabled:
        print("Email Alerts: " + vp.color("Enabled", "green"))
    else:
        print("Email Alerts: " + vp.color("Disabled", "red"))
    print("Max Restarts: " + vp.color(f"{max_restarts}", "yellow"))
    if restart_after_delay is not None:
        interval_minutes = restart_after_delay / 60.0
        print(
            "Scheduled Restart Interval: "
            + vp.color(f"{interval_minutes:g} minutes", "yellow")
        )
    if scheduled_restart_memory_threshold_bytes is not None:
        threshold_gib = scheduled_restart_memory_threshold_bytes / float(1024**3)
        poll_minutes = scheduled_restart_poll_interval_seconds / 60.0
        print(
            "Memory-Aware Scheduled Restart Threshold: "
            + vp.color(f"{threshold_gib:g} GiB", "yellow")
        )
        print(
            "Memory-Aware Scheduled Restart Polling: "
            + vp.color(f"{poll_minutes:g} minutes", "yellow")
        )
    if detect_memory_leaks:
        print("Memory Leak Detection: " + vp.color("Enabled", "green"))
        print(
            "Memory Leak Warm-up: "
            + vp.color(f"{memory_leak_warmup_seconds / 60.0:g} minutes", "yellow")
        )
    else:
        print("Memory Leak Detection: " + vp.color("Disabled", "red"))
    print("=" * 80)
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


def print_completion_summary(
    restart_count: int,
    total_runtime: Optional[float] = None,
    scheduled_restart_count: int = 0,
) -> None:
    """Print the final restart counts for a monitored process.

    Args:
        restart_count (int): Number of genuine crash or failure restarts.
        total_runtime (Optional[float]): Total monitored runtime in seconds.
            The current compact summary does not display this value.
        scheduled_restart_count (int): Number of successful scheduled
            restarts. These restarts are displayed separately and are not part
            of ``restart_count``. Defaults to ``0``.

    Returns:
        None: The function prints the summary to stdout.

    Raises:
        RuntimeError: Not raised by this helper.
    """
    # print("\n" + "=" * 50)
    print("\n" + vp.color("Process Completed", "green"))
    # print("=" * 50)
    print(
        "Total Restarts: "
        + vp.color(f"{restart_count + scheduled_restart_count}", "orange")
    )
    print("Crash/Failure Restarts: " + vp.color(f"{restart_count}", "orange"))
    print("Scheduled Restarts: " + vp.color(f"{scheduled_restart_count}", "orange"))
    # if total_runtime is not None:
    #     print(f"Total Runtime:  {total_runtime:.1f}s")
    # print("=" * 50)


def print_error_message(error_type: str, message: str) -> None:
    """Print error messages with consistent formatting."""
    vp.printf(f"ERROR [{error_type}]: {message}", color="red")


def print_warning_message(message: str) -> None:
    """Print warning messages with consistent formatting."""
    vp.printf(f"WARNING: {message}", color="yellow")


def print_success_message(message: str) -> None:
    """Print success messages with consistent formatting."""
    vp.printf(f"SUCCESS: {message}", color="green")


def print_cleanup_info(terminated: int, killed: int) -> None:
    """Print child process cleanup information."""
    if terminated > 0 or killed > 0:
        vp.printf(f"Child cleanup: {terminated} terminated, {killed} killed", color="yellow")


# —————————————————————————————————— Utility ————————————————————————————————— #
def _cleanup_stale_monitor_files():
    """Remove orphaned monitor helper files from the temporary directory."""
    tmpdir = tempfile.gettempdir()
    for path in glob.glob(os.path.join(tmpdir, "*_monitor.*")):
        try:
            os.unlink(path)
        except OSError:
            pass


def get_process_resource_usage(pid: int) -> Tuple[float, float, float]:
    """Return process-tree memory percentage, memory in GB and CPU percentage.

    This helper queries ``psutil`` for the resource consumption of the process
    tree rooted at ``pid``. The CPU value reflects the sum across all
    available CPU cores and therefore may exceed ``100`` when the monitored job
    utilises more than one core across multiple child processes. Measuring the
    full process tree prevents wrapper shells or helper children from making
    the displayed values look artificially close to zero.

    Args:
        pid (int): Root process ID of the monitored job tree to query.

    Returns:
        Tuple[float, float, float]:
            The memory usage percentage, memory usage in gigabytes and CPU
            percentage for the given process tree.

    Raises:
        psutil.NoSuchProcess: If the PID does not exist.
    """
    proc = psutil.Process(pid)

    # Sampling the full process tree keeps resource numbers attached to the
    # monitored workload instead of whichever wrapper shell happened to be
    # launched by the terminal helper.
    try:
        processes = [proc] + proc.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.Error, OSError):
        processes = [proc]

    for process in processes:
        try:
            process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.Error, OSError):
            continue

    time.sleep(1.0)

    total_rss_bytes = 0
    cpu_percent = 0.0
    for process in processes:
        try:
            with process.oneshot():
                total_rss_bytes += process.memory_info().rss
                cpu_percent += process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.Error, OSError):
            continue

    memory_total = psutil.virtual_memory().total
    mem_percent = (total_rss_bytes / memory_total) * 100.0 if memory_total else 0.0
    mem_gb = total_rss_bytes / (1024**3)

    # Write usage information to a log file if configured
    if RESOURCE_USAGE_LOG_FILE:
        try:
            with open(RESOURCE_USAGE_LOG_FILE, "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
                f.write(
                    f"Time: {timestamp}, PID: {pid}, MEM: {mem_percent:.2f}%, {mem_gb:.2f} GB, CPU: {cpu_percent:.2f}%\n"
                )
        except Exception:
            pass

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


def start_monitor(pid: int, title: str, supress_tf_warnings: bool = False) -> Dict[str, Any]:
    """Start simplified crash monitor without email capabilities.

    Args:
        pid (int): Process ID to monitor.
        title (str): Process title for alerts.
        supress_tf_warnings (bool): When ``True``, the caller is asking to
            suppress TensorFlow warnings for the monitored target process. This
            helper monitor does not execute TensorFlow code, so the value does
            not change helper-monitor behavior. When ``False``, the helper
            still launches identically. The parameter remains in the signature
            to preserve the public runtime call chain.

    Returns:
        Dict[str, Any]: Monitor control info dictionary containing the helper
        process handle and the temporary control file paths used to coordinate
        crash detection and shutdown.

    Raises:
        ValueError: If ``pid`` does not exist when monitoring begins.
        OSError: If the helper monitor process fails to start.

    Examples:
        >>> info = start_monitor(12345, "Training Job")
        >>> sorted(info.keys())  # doctest: +ELLIPSIS
        ['pid_file', 'process', 'restart_file', 'script_path', 'stop_file']
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

    # The helper monitor should stay invisible. Only the target process needs a
    # user-facing terminal window, while this watchdog should run quietly in the
    # background and surface errors only if startup fails.
    process = subprocess.Popen(
        [sys.executable, script_path],
        cwd=os.getcwd(),
        start_new_session=False,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=False,
    )

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
        monitor_info (Dict[str, Any]): Monitor control info from start_monitor()
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
        monitor_info (Dict[str, Any]): Monitor control info

    Returns:
        Dict[str, Any]: Dictionary with crash info or empty dict if no crash
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
    success_flag_file: str = DEFAULT_SUCCESS_FLAG_FILE,
    title: Optional[str] = None,
    max_restarts: int = 10,
    restart_delay: float = 3.0,
    recipients_file: Optional[str] = None,
    credentials_file: Optional[str] = None,
    force_restart: Optional[float] = None,
    scheduled_restart_memory_threshold_bytes: Optional[float] = None,
    scheduled_restart_poll_interval_seconds: Optional[float] = None,
    retry_attempts: int = None,
    supress_tf_warnings: bool = False,
    resource_usage_log_file: Optional[str] = None,
    restart_email_warning: bool = True,
    report_logs: bool = False,
    detect_memory_leaks: bool = False,
    memory_leak_warmup_seconds: float = 300.0,
) -> None:
    """Main function with notebook conversion, file cleanup, and consolidated email notification support.

    The function validates the existence of ``file_path`` before allocating
    monitoring resources. If the path does not point to a valid file a
    ``FileNotFoundError`` is raised immediately.

    Args:
        file_path (str): Path to .py or .ipynb file to execute
        success_flag_file (str): Path to success flag file
        title (Optional[str]): Custom title for monitoring and email alerts
        max_restarts (int): Maximum restart attempts
        restart_delay (float): Delay between restarts in seconds
        recipients_file (Optional[str]): Path to recipients JSON file (defaults to ./json/recipients.json)
        credentials_file (Optional[str]): Path to credentials JSON file (defaults to ./json/credentials.json)
        force_restart (Optional[float]): Positive fixed interval in seconds for
            scheduled restarts. ``None`` disables the fixed policy. Scheduled
            restarts do not consume the genuine-crash restart budget.
        scheduled_restart_memory_threshold_bytes (Optional[float]): Positive
            process-tree RSS threshold in bytes for smart scheduled restarts.
            When set with ``scheduled_restart_poll_interval_seconds``, the
            monitor checks RSS at the polling interval and restarts when RSS
            reaches this threshold. ``None`` preserves fixed scheduled
            restarts.
        scheduled_restart_poll_interval_seconds (Optional[float]): Positive
            interval in seconds between smart scheduled-restart RSS checks.
            It must be set together with
            ``scheduled_restart_memory_threshold_bytes``.
        retry_attempts (int): Number of retry attempts before sending failure email
        supress_tf_warnings (bool): Suppress TensorFlow warnings (default: False)
        resource_usage_log_file (Optional[str]): Path to write process resource usage logs. If None, logging is disabled.
        restart_email_warning (bool): Enable or disable email warnings for restart events
        report_logs (bool): When ``True``, write monitor report artifacts under
            ``runs/monitor_logs`` for each monitored target. If ``False``, the
            runtime preserves the previous no-report behavior.
        detect_memory_leaks (bool): When ``True``, monitor process-tree RSS for
            a conservative possible-leak trend. Detection only warns and does
            not stop or restart the target. Defaults to ``False``.
        memory_leak_warmup_seconds (float): Positive finite initial duration in
            seconds excluded from memory leak analysis. Samples from this
            period remain visible in the warning graph. Defaults to ``300.0``.

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is unsupported or smart scheduled-restart
            configuration is incomplete or invalid.
        ImportError: If notebook dependencies missing for .ipynb files
    """

    global RESOURCE_USAGE_LOG_FILE
    RESOURCE_USAGE_LOG_FILE = resource_usage_log_file

    # Validate that the requested file exists before initializing monitoring
    FileTypeHandler.validate_file(file_path)

    resolved_success_flag = _resolve_success_flag_file(file_path, success_flag_file)

    try:
        # late import to avoid circular dependencies
        from .restart_manager import FlagBasedRestartManager

        # Clean up any existing success flag file before starting
        Path(resolved_success_flag).unlink(missing_ok=True)

        manager = FlagBasedRestartManager(
            max_restarts=max_restarts,
            restart_delay=restart_delay,
            recipients_file=recipients_file,
            credentials_file=credentials_file,
            retry_attempts=max_restarts if retry_attempts is None else retry_attempts,
            restart_email_warning=restart_email_warning,
            report_logs=report_logs,
            scheduled_restart_memory_threshold_bytes=scheduled_restart_memory_threshold_bytes,
            scheduled_restart_poll_interval_seconds=scheduled_restart_poll_interval_seconds,
            detect_memory_leaks=detect_memory_leaks,
            memory_leak_warmup_seconds=memory_leak_warmup_seconds,
        )

        manager.run_file_with_restart(
            file_path=file_path,
            success_flag_file=resolved_success_flag,
            title=title,
            restart_after_delay=force_restart,
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


def _resolve_success_flag_file(file_path: str, success_flag_file: str) -> str:
    """Return a collision-free success flag file path.

    Detailed Description:
        The monitoring system historically defaulted to ``/tmp/success.flag``
        for the completion marker written by monitored processes. When
        multiple monitors executed simultaneously on the same machine, this
        shared default path caused monitors to observe each other's completion
        flags and stop prematurely. This helper generates a unique success flag
        path whenever the legacy default is requested, preventing unintended
        cross-monitor interference while preserving user supplied paths.

    Args:
        file_path (str): Absolute or relative path to the file being
            monitored. The stem of this path is incorporated into the
            generated flag name for easier debugging.
        success_flag_file (str): Requested path for the success flag. When this
            value matches the legacy default ``/tmp/success.flag``, a unique
            path is generated inside the system temporary directory.

    Returns:
        str: The original success flag path when it is custom, otherwise a
        unique path located within the system temporary directory.

    Raises:
        ValueError: If ``file_path`` or ``success_flag_file`` are empty.

    Notes:
        The generated filenames follow the pattern
        ``success_<sanitized_stem>_<uuid>.flag``. Sanitization replaces any
        non-alphanumeric characters with underscores to ensure compatibility
        across filesystems.
    """

    if not file_path:
        raise ValueError("file_path must be provided to resolve success flag file")
    if not success_flag_file:
        raise ValueError(
            "success_flag_file must be provided to resolve success flag file"
        )

    if Path(success_flag_file).resolve() != Path(DEFAULT_SUCCESS_FLAG_FILE).resolve():
        return success_flag_file

    sanitized_stem = Path(file_path).stem or "job"
    sanitized_stem = "".join(
        character if character.isalnum() else "_" for character in sanitized_stem
    )
    sanitized_stem = sanitized_stem[:32] or "job"
    unique_name = f"success_{sanitized_stem}_{uuid.uuid4().hex}.flag"
    unique_path = Path(tempfile.gettempdir()) / unique_name
    return str(unique_path)
